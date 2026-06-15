import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 
# STEP 1: Reading the Dataset
#

DATA_PATH = "data_refined.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values per column:")
print(df.isna().sum())

# Drop missing values if any exist
df = df.dropna()

print("\nShape after dropping missing values:")
print(df.shape)


# 
# STEP 2: Feature Selection
# Target column: Diagnosed
# M = Malignant tumor
# B = Benign tumor
# 

target = "Diagnosed"

print("\nTarget distribution before encoding:")
print(df[target].value_counts())

# Convert target labels to numbers
# M = malignant = 1
# B = benign = 0
df[target] = df[target].map({
    "M": 1,
    "B": 0
})

print("\nTarget distribution after encoding:")
print(df[target].value_counts())


# Keep only numeric columns for correlation and modeling
numeric_df = df.select_dtypes(include=["int64", "float64"])

# Remove ID columns if they exist
possible_id_columns = ["id", "ID", "Id"]
for col in possible_id_columns:
    if col in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[col])

# Calculate correlation of features with target
correlations = numeric_df.corr()[target].drop(target).abs().sort_values(ascending=False)

print("\nCorrelation of features with target:")
print(correlations)

# Choose important features above a correlation limit
correlation_limit = 0.40
important_features = correlations[correlations >= correlation_limit].index.tolist()

print("\nImportant features selected using correlation limit:")
print(important_features)

print("\nNumber of important features:")
print(len(important_features))

# Full feature set
full_features = [col for col in numeric_df.columns if col != target]

# Reduced feature set
reduced_features = important_features

X_full = df[full_features]
X_reduced = df[reduced_features]
y = df[target]

print("\nNumber of full features:")
print(len(full_features))

print("\nNumber of reduced features:")
print(len(reduced_features))


# 
# STEP 3: Splitting the Data
# 80% training set
# 10% validation set
# 10% test set
# 

def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


X_train_full, X_val_full, X_test_full, y_train, y_val, y_test = split_data(X_full, y)

X_train_red, X_val_red, X_test_red, y_train_red, y_val_red, y_test_red = split_data(X_reduced, y)

print("\nFull features training set:")
print(X_train_full.shape, y_train.shape)

print("\nFull features validation set:")
print(X_val_full.shape, y_val.shape)

print("\nFull features test set:")
print(X_test_full.shape, y_test.shape)

print("\nReduced features training set:")
print(X_train_red.shape, y_train_red.shape)

print("\nReduced features validation set:")
print(X_val_red.shape, y_val_red.shape)

print("\nReduced features test set:")
print(X_test_red.shape, y_test_red.shape)


# 
# Helper function for model evaluation
# 

def evaluate_model(model_name, model, X_train, X_val, X_test, y_train, y_val, y_test):
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print("\n====================================================")
    print(model_name)
    print("====================================================")

    print("Validation Accuracy:")
    print(round(val_accuracy, 4))

    print("Test Accuracy:")
    print(round(test_accuracy, 4))

    print("\nConfusion Matrix:")
    print("Labels: 0 = Benign, 1 = Malignant")
    print(confusion_matrix(y_test, y_test_pred, labels=[0, 1]))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_test_pred,
        target_names=["Benign", "Malignant"]
    ))

    return {
        "model": model_name,
        "validation_accuracy": val_accuracy,
        "test_accuracy": test_accuracy
    }


# 
# STEP 4: Training Classifiers
# KNN, Random Forest, and SVC
# Train both full feature set and reduced feature set
# 

results = []


#
# KNN: choose optimal k using cross-validation
# 

def find_best_k(X_train, y_train):
    k_values = list(range(1, 31, 2))
    cv_scores = []

    for k in k_values:
        knn_pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=k))
            ]
        )

        scores = cross_val_score(
            knn_pipeline,
            X_train,
            y_train,
            cv=5,
            scoring="accuracy"
        )

        mean_score = scores.mean()
        cv_scores.append(mean_score)

        print(f"k = {k:2d} | Cross-validation accuracy = {mean_score:.4f}")

    best_index = int(np.argmax(cv_scores))
    best_k = k_values[best_index]

    return best_k


print("\nFinding best k for KNN using full features:")
best_k_full = find_best_k(X_train_full, y_train)

print("\nBest k for full features:")
print(best_k_full)

knn_full = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=best_k_full))
    ]
)

results.append(
    evaluate_model(
        "KNN - Full Features",
        knn_full,
        X_train_full,
        X_val_full,
        X_test_full,
        y_train,
        y_val,
        y_test
    )
)


print("\nFinding best k for KNN using reduced features:")
best_k_reduced = find_best_k(X_train_red, y_train_red)

print("\nBest k for reduced features:")
print(best_k_reduced)

knn_reduced = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=best_k_reduced))
    ]
)

results.append(
    evaluate_model(
        "KNN - Reduced Features",
        knn_reduced,
        X_train_red,
        X_val_red,
        X_test_red,
        y_train_red,
        y_val_red,
        y_test_red
    )
)


# 
# Random Forest
# 

rf_full = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

results.append(
    evaluate_model(
        "Random Forest - Full Features",
        rf_full,
        X_train_full,
        X_val_full,
        X_test_full,
        y_train,
        y_val,
        y_test
    )
)


rf_reduced = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

results.append(
    evaluate_model(
        "Random Forest - Reduced Features",
        rf_reduced,
        X_train_red,
        X_val_red,
        X_test_red,
        y_train_red,
        y_val_red,
        y_test_red
    )
)


#
# Support Vector Classifier
# 

svc_full = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42))
    ]
)

results.append(
    evaluate_model(
        "SVC - Full Features",
        svc_full,
        X_train_full,
        X_val_full,
        X_test_full,
        y_train,
        y_val,
        y_test
    )
)


svc_reduced = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42))
    ]
)

results.append(
    evaluate_model(
        "SVC - Reduced Features",
        svc_reduced,
        X_train_red,
        X_val_red,
        X_test_red,
        y_train_red,
        y_val_red,
        y_test_red
    )
)


# 
# Compare Results
# 

results_df = pd.DataFrame(results)

print("\n====================================================")
print("Final Comparison")
print("====================================================")

print(results_df.sort_values(by="test_accuracy", ascending=False))

best_model = results_df.loc[results_df["test_accuracy"].idxmax()]

print("\nBest model:")
print(best_model)

print("\nMinimum accuracy requirement:")
print("The project requires at least 94% accuracy.")

if best_model["test_accuracy"] >= 0.94:
    print("Requirement satisfied: accuracy is at least 94%.")
else:
    print("Requirement not satisfied: try changing the correlation limit or tuning models.")


# 
# STEP 5: Challenge Yourself
# Another way to reduce features: Random Forest importance
# 

print("\n====================================================")
print("Challenge: Feature Reduction Using Random Forest Importance")
print("====================================================")

rf_feature_selector = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

rf_feature_selector.fit(X_train_full, y_train)

feature_importance = pd.Series(
    rf_feature_selector.feature_importances_,
    index=full_features
).sort_values(ascending=False)

print("\nRandom Forest feature importance:")
print(feature_importance)

top_n = 10
rf_selected_features = feature_importance.head(top_n).index.tolist()

print(f"\nTop {top_n} features selected by Random Forest importance:")
print(rf_selected_features)

X_rf_selected = df[rf_selected_features]

X_train_rf_sel, X_val_rf_sel, X_test_rf_sel, y_train_rf_sel, y_val_rf_sel, y_test_rf_sel = split_data(
    X_rf_selected,
    y
)

svc_rf_selected = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42))
    ]
)

challenge_result = evaluate_model(
    "SVC - Random Forest Selected Features",
    svc_rf_selected,
    X_train_rf_sel,
    X_val_rf_sel,
    X_test_rf_sel,
    y_train_rf_sel,
    y_val_rf_sel,
    y_test_rf_sel
)

print("\nChallenge result:")
print(challenge_result)
