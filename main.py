import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from scipy.stats import spearmanr
from lime.lime_tabular import LimeTabularExplainer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")


# First I load the data
df = pd.read_csv("ai_skepticism_dataset.csv")
target_col = "user_skepticism_category"

X = df.drop(columns=[target_col])
y = df[target_col]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_.tolist()

# Deciding on the column types
bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist() + bool_cols
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Managing train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Preprocessing
numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])
categorical_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),("onehot", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([("num", numeric_transformer, num_cols), ("cat", categorical_transformer, cat_cols),])

# Modeling
rf_pipeline = Pipeline([("preprocessor", preprocessor),("classifier", RandomForestClassifier(n_estimators=200, random_state=42))])

rf_pipeline.fit(X_train, y_train)
y_pred = rf_pipeline.predict(X_test)

print("Random Forest Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Random Forest Macro F1:", round(f1_score(y_test, y_pred, average="macro"), 4))

# Transforming data for SHAP and LIME
fitted_preprocessor = rf_pipeline.named_steps["preprocessor"]
rf_model = rf_pipeline.named_steps["classifier"]

X_train_transformed = fitted_preprocessor.transform(X_train)
X_test_transformed = fitted_preprocessor.transform(X_test)

if hasattr(X_train_transformed, "toarray"): 
    X_train_transformed = X_train_transformed.toarray()
if hasattr(X_test_transformed, "toarray"): 
    X_test_transformed = X_test_transformed.toarray()

feature_names = fitted_preprocessor.get_feature_names_out()

X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=feature_names)
X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names)

# Implementing the SHAP explanation
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_transformed_df)

if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3: shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

pred_class = int(rf_model.predict(X_test_transformed_df.iloc[[0]])[0])

print("\nSample SHAP top features:")
abs_vals = np.abs(shap_values[pred_class][0])
top_idx = np.argsort(abs_vals)[-5:][::-1]
for i in top_idx:
     print(feature_names[i], "->", round(shap_values[pred_class][0][i], 4))

plt.figure()
shap.summary_plot( shap_values[pred_class], X_test_transformed_df, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=300, bbox_inches="tight")
plt.close()


# Implementing LIME explanation
lime_explainer = LimeTabularExplainer( training_data=X_train_transformed_df.values, feature_names=feature_names.tolist(), class_names=class_names, mode="classification", discretize_continuous=True, random_state=42)

lime_exp = lime_explainer.explain_instance(data_row=X_test_transformed_df.iloc[0].values, predict_fn=rf_model.predict_proba, num_features=5)

print("\nSample LIME explanation:")
for item in lime_exp.as_list():
    print(item)

lime_exp.save_to_file("lime_explanation.html")


# Function for the faithfulness score
def faithfulness_score(model, x_instance, important_indices):
    original_probs = model.predict_proba(x_instance.reshape(1, -1))[0]
    original_class = int(np.argmax(original_probs))
    original_conf = float(original_probs[original_class])

    x_modified = x_instance.copy()
    for idx in important_indices:
        x_modified[idx] = 0 # Neutralize the top-k most important SHAP features by setting them to 0.

    new_probs = model.predict_proba(x_modified.reshape(1, -1))[0]
    new_conf = float(new_probs[original_class])

    return original_conf - new_conf


shap_faithfulness = []
for i in range(min(20, len(X_test_transformed_df))):
    pred_class = int(rf_model.predict(X_test_transformed_df.iloc[[i]])[0])
    abs_vals = np.abs(shap_values[pred_class][i])
    top_idx = np.argsort(abs_vals)[-3:][::-1]
    score = faithfulness_score(rf_model, X_test_transformed_df.iloc[i].values, top_idx)
    shap_faithfulness.append(score)

print("\nAverage SHAP faithfulness:", round(float(np.mean(shap_faithfulness)), 4))


# Stability
def add_small_noise(x, noise_std=0.01):
    return x + np.random.normal(0, noise_std, size=x.shape)


shap_stability = []
for i in range(min(20, len(X_test_transformed_df))):
    x_orig = X_test_transformed_df.iloc[i].values
    x_noisy = add_small_noise(x_orig)

    x_orig_df = pd.DataFrame([x_orig], columns=feature_names)
    x_noisy_df = pd.DataFrame([x_noisy], columns=feature_names)

    shap_orig = explainer.shap_values(x_orig_df)
    shap_noisy = explainer.shap_values(x_noisy_df)

    if isinstance(shap_orig, np.ndarray) and shap_orig.ndim == 3:
        shap_orig = [shap_orig[:, :, j] for j in range(shap_orig.shape[2])]
    if isinstance(shap_noisy, np.ndarray) and shap_noisy.ndim == 3:
        shap_noisy = [shap_noisy[:, :, j] for j in range(shap_noisy.shape[2])]

    pred_class_orig = int(rf_model.predict(x_orig.reshape(1, -1))[0])
    pred_class_noisy = int(rf_model.predict(x_noisy.reshape(1, -1))[0])

    corr, _ = spearmanr(
        np.abs(shap_orig[pred_class_orig][0]),
        np.abs(shap_noisy[pred_class_noisy][0]),
    )
    if not np.isnan(corr):
        shap_stability.append(corr)

print("Average SHAP stability:", round(float(np.mean(shap_stability)), 4))

lr_pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000, random_state=42))])
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)
print("Logistic Regression Accuracy:", round(accuracy_score(y_test, y_pred_lr), 4))
print("Logistic Regression Macro F1:", round(f1_score(y_test, y_pred_lr, average="macro"), 4))
 
 
# Helper: dense LIME weight vector aligned to feature indices
def lime_weight_vector(lime_exp, n_features):
    weight_map = dict(lime_exp.as_map()[lime_exp.available_labels()[0]])
    vec = np.zeros(n_features)
    for idx, w in weight_map.items():
        vec[idx] = w
    return vec
 
 
# Faithfulness — insertion test (complement to deletion already implemented above)
def faithfulness_insertion(model, x_instance, important_indices):
    orig_class = int(np.argmax(model.predict_proba(x_instance.reshape(1, -1))[0]))
    x_restored = np.zeros_like(x_instance)
    for idx in important_indices:
        x_restored[idx] = x_instance[idx]
    restored_conf = float(model.predict_proba(x_restored.reshape(1, -1))[0][orig_class])
    baseline_conf = float(model.predict_proba(np.zeros_like(x_instance).reshape(1, -1))[0][orig_class])
    return restored_conf - baseline_conf
 
 
# F-Fidelity (marginal baseline instead of zeros to avoid out-of-distribution artifacts)
feature_means = X_train_transformed_df.mean().values.copy()
feature_stds  = X_train_transformed_df.std().values.copy()
feature_stds[feature_stds == 0] = 1e-8
 
def f_fidelity(model, x_instance, important_indices, B=10):
    orig_probs = model.predict_proba(x_instance.reshape(1, -1))[0]
    orig_class = int(np.argmax(orig_probs))
    drops = []
    for _ in range(B):
        x_mod = x_instance.copy()
        for idx in important_indices:
            x_mod[idx] = np.random.normal(feature_means[idx], feature_stds[idx])
        drops.append(float(orig_probs[orig_class]) - float(model.predict_proba(x_mod.reshape(1, -1))[0][orig_class]))
    return float(np.mean(drops))
 
 
# Evaluating all metrics for SHAP and LIME
shap_ins, lime_del, lime_ins, shap_ff, lime_ff, lime_stability = [], [], [], [], [], []
 
for i in range(min(20, len(X_test_transformed_df))):
    x_val      = X_test_transformed_df.iloc[i].values
    pred_class = int(rf_model.predict(x_val.reshape(1, -1))[0])
    n_feat     = len(feature_names)
 
    shap_top = np.argsort(np.abs(shap_values[pred_class][i]))[-3:][::-1]
    shap_ins.append(faithfulness_insertion(rf_model, x_val, shap_top))
    shap_ff.append(f_fidelity(rf_model, x_val, shap_top))
 
    lime_exp = lime_explainer.explain_instance(x_val, rf_model.predict_proba, num_features=3, labels=(pred_class,))
    lime_top = np.argsort(np.abs(lime_weight_vector(lime_exp, n_feat)))[-3:][::-1]
    lime_del.append(faithfulness_score(rf_model, x_val, lime_top))
    lime_ins.append(faithfulness_insertion(rf_model, x_val, lime_top))
    lime_ff.append(f_fidelity(rf_model, x_val, lime_top))
 
    # LIME stability
    x_noisy   = add_small_noise(x_val)
    le_orig   = lime_explainer.explain_instance(x_val,   rf_model.predict_proba, num_features=n_feat, labels=(pred_class,))
    pc_noisy  = int(rf_model.predict(x_noisy.reshape(1, -1))[0])
    le_noisy  = lime_explainer.explain_instance(x_noisy, rf_model.predict_proba, num_features=n_feat, labels=(pc_noisy,))
    corr, _   = spearmanr(np.abs(lime_weight_vector(le_orig, n_feat)), np.abs(lime_weight_vector(le_noisy, n_feat)))
    if not np.isnan(corr):
        lime_stability.append(corr)
 
 
# SHAP additivity check
additivity_errors = []
for i in range(min(20, len(X_test_transformed_df))):
    x_val      = X_test_transformed_df.iloc[[i]]
    pred_class = int(rf_model.predict(x_val)[0])
    sv_i = explainer.shap_values(x_val)
    if isinstance(sv_i, np.ndarray) and sv_i.ndim == 3:
        sv_i = [sv_i[:, :, j] for j in range(sv_i.shape[2])]
    ev      = float(explainer.expected_value[pred_class]) if hasattr(explainer.expected_value, "__len__") else float(explainer.expected_value)
    f_x     = float(rf_model.predict_proba(x_val)[0][pred_class])
    additivity_errors.append(abs(np.sum(sv_i[pred_class][0]) + ev - f_x))
 
print("Average SHAP additivity error:", round(float(np.mean(additivity_errors)), 6))
 
# Summary table for comparison
results = pd.DataFrame({
    "Method": ["SHAP", "LIME"],
    "Faithfulness (deletion)":  [round(float(np.mean(shap_faithfulness)), 4), round(float(np.mean(lime_del)), 4)],
    "Faithfulness (insertion)": [round(float(np.mean(shap_ins)), 4),          round(float(np.mean(lime_ins)), 4)],
    "F-Fidelity":               [round(float(np.mean(shap_ff)), 4),           round(float(np.mean(lime_ff)), 4)],
    "Stability (Spearman)":     [round(float(np.mean(shap_stability)), 4),    round(float(np.mean(lime_stability)), 4)],
}).set_index("Method")

 
# Comparison bar chart
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, (col, color) in zip(axes, zip(results.columns, ["#4C72B0", "#4C72B0", "#4C72B0", "#4C72B0"])):
    vals = results[col].values
    bars = ax.bar(results.index, vals, color=["#4C72B0", "#DD8452"], width=0.45)
    ax.set_title(col, fontsize=9)
    ax.set_ylim(0, max(vals) * 1.4 if max(vals) > 0 else 1)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
plt.suptitle("SHAP vs LIME across evaluation dimensions", fontsize=11)
plt.tight_layout()
plt.savefig("xai_comparison.png", dpi=150, bbox_inches="tight")
plt.close()


# Implementing Multi-dimensional concept discovery (MCD) 

N_CONCEPTS = len(class_names)

# Leaf embeddings
leaf_train = rf_model.apply(X_train_transformed_df.values).astype(float)
leaf_test  = rf_model.apply(X_test_transformed_df.values).astype(float)

leaf_train_s = StandardScaler().fit_transform(leaf_train)
leaf_test_s  = StandardScaler().fit(leaf_train).transform(leaf_test)

# Clustering
clusters = KMeans(n_clusters=N_CONCEPTS, random_state=42, n_init=10).fit_predict(leaf_train_s)

# Build concept bases (PCA per cluster)
def build_concepts(X, labels):
    bases, dims = {}, {}
    for c in range(N_CONCEPTS):
        members = X[labels == c]
        if len(members) < 2:
            bases[c], dims[c] = None, 0
            continue
        pca = PCA(random_state=42).fit(members)
        d = np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.9) + 1
        bases[c] = PCA(n_components=min(d, members.shape[1]), random_state=42).fit(members)
        dims[c] = bases[c].n_components_
    return bases, dims

concept_bases, concept_dims = build_concepts(leaf_train_s, clusters)

print("MCD concept subspace dimensions:")
for c in range(N_CONCEPTS):
    print(f"  {class_names[c]}: d={concept_dims[c]}")

# Projection helper
project = lambda v, p: np.zeros_like(v) if p is None else p.components_.T @ (p.components_ @ v)

# Completeness + relevance
scores, rels = [], []
for v in leaf_test_s:
    norm = v @ v
    if norm == 0:
        scores.append(0); rels.append([0]*N_CONCEPTS); continue
    proj_vals = [float((p:=project(v, concept_bases[c])) @ p) for c in range(N_CONCEPTS)]
    scores.append(sum(proj_vals) / norm)
    rels.append([pv / norm for pv in proj_vals])

mcd_completeness = float(np.mean(scores))
mean_rel = np.mean(rels, axis=0)

print(f"\nMCD completeness (η): {round(mcd_completeness, 4)}")
for c in range(N_CONCEPTS):
    print(f"  {class_names[c]}: {round(float(mean_rel[c]), 4)}")

# Plot
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].bar(class_names, mean_rel)
ax[0].set_title("Concept relevance")

ax[1].hist(scores, bins=20)
ax[1].axvline(mcd_completeness, linestyle="--", color="red")
ax[1].set_title("Completeness distribution")

plt.tight_layout()
plt.savefig("mcd_concept_analysis.png", dpi=150)
plt.close()

print("Saved: mcd_concept_analysis.png")
