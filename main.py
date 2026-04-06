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

if hasattr(X_train_transformed, "toarray"): X_train_transformed = X_train_transformed.toarray()
if hasattr(X_test_transformed, "toarray"): X_test_transformed = X_test_transformed.toarray()

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


# 9. Function for the faithfulness score
def faithfulness_score(model, x_instance, important_indices):
    original_probs = model.predict_proba(x_instance.reshape(1, -1))[0]
    original_class = int(np.argmax(original_probs))
    original_conf = float(original_probs[original_class])

    x_modified = x_instance.copy()
    for idx in important_indices:
        x_modified[idx] = 0

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
