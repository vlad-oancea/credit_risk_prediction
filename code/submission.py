import os
import warnings
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
)

from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.metrics import AUC  # type: ignore


tf.config.optimizer.set_jit(True)  # Enable XLA
os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Download the dataset
os.makedirs("data", exist_ok=True)
data_file = "data/accepted_2007_to_2018Q4.csv"
if not os.path.exists(data_file):
    dataset_url = "https://github.com/vlad-oancea/credit_risk_prediction/releases/download/dataset-v1/accepted_2007_to_2018Q4.csv"
    with open(data_file, "wb") as file:
        file.write(requests.get(dataset_url).content)

# Load the dataset
df_orig = pd.read_csv(data_file, sep=",")

# =========================================================================================================================================
#                                                           DATA CLEANING
# =========================================================================================================================================

# Keep the relevant columns
characteristics = [
    "id",
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "sub_grade",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "issue_d",
    "loan_status",
    "purpose",
    "dti",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
]
df = df_orig[characteristics]

# Simplify the Home ownership column
df.loc[df["home_ownership"].isin(["ANY", "NONE"]), "home_ownership"] = "OTHER"

# Keep only loans that are either paid or defaulted, get rid of current ones
df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])]

# Create the column with quarters instead of month from issue_d (to have less dummies and simplify computation)
df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
df["issue_q"] = df["issue_d"].dt.to_period("Q")
df = df.drop(["issue_d"], axis=1)

# Ensure lines with missing values are less than 5% of the data, and remove them because we have already a lot of lines
for column in df.columns:
    if df[column].isna().sum() != 0:
        missing = df[column].isna().sum()
        portion = (missing / df.shape[0]) * 100
        print(
            f"For column '{column}' there are {missing} missing values, i.e. {portion:.3f}%"
        )
df = df.dropna()


# Remove outliers
# We decided not to run it on loan_amnt and int_rate as they are not as subject to ouliers and the code was removing unnecessary lines.
outlier_columns = [
    "annual_inc",
    "dti",
    "installment",
    "open_acc",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
]
reduced_df = df.copy()

for col in outlier_columns:
    col_zscore = (df[col] - df[col].mean()) / df[col].std()
    reduced_df = reduced_df[np.abs(col_zscore) < 3]

# We decided to filter outliers manually for the columns pub_rec and pub_rec_bankruptcies to avoid removing too many lines.
reduced_df = reduced_df[reduced_df["pub_rec"] <= 15]
reduced_df = reduced_df[reduced_df["pub_rec_bankruptcies"] <= 7]

reduced_df = reduced_df.set_index("id")

# Create the dummy variables
dummies = reduced_df.select_dtypes(include="object").columns.tolist()
dummies = [col for col in dummies if col != "loan_status"]
dummies = dummies + ["issue_q"]
cleaned_df = pd.get_dummies(reduced_df, columns=dummies, drop_first=True)

# =========================================================================================================================================
#                                                     CREATE RESAMPLED DATASETS
# =========================================================================================================================================

# Assign binary values to the target variable
cleaned_df["loan_status"] = cleaned_df["loan_status"].map(
    {"Fully Paid": 1, "Charged Off": 0}
)

# Split into training and testing sets (80/20 split)
X = cleaned_df.drop(columns=["loan_status"], axis=1)
y = cleaned_df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Aggregate the test dataset
test_data = X_test.copy()
test_data["loan_status"] = y_test


# Aggregate the first training dataset : the original data
train_data_original = X_train.copy()
train_data_original["loan_status"] = y_train


# Create the second training dataset : the undersampled data (using Tomek Links and random undersampling)
tomek = TomekLinks(sampling_strategy="auto")
X_tomek, y_tomek = tomek.fit_resample(X_train, y_train)

class_distribution = np.bincount(y_tomek)
desired_minority_count = class_distribution[0]
desired_majority_count = desired_minority_count
rus = RandomUnderSampler(
    sampling_strategy={0: desired_majority_count, 1: desired_minority_count},
    random_state=42,
)
X_undersampled, y_undersampled = rus.fit_resample(X_tomek, y_tomek)

train_data_undersampled = X_undersampled.copy()
train_data_undersampled["loan_status"] = y_undersampled


# Create the third training dataset : the oversampled data
count_paid = (train_data_original["loan_status"] == 1).sum()
count_default = (train_data_original["loan_status"] == 0).sum()
num_additional_zeros = count_paid - count_default

zeros_to_add = train_data_original[train_data_original["loan_status"] == 0].sample(
    n=num_additional_zeros, replace=True
)
train_data_oversampled = pd.concat(
    [train_data_original, zeros_to_add], ignore_index=True
)


# Create the fourth training dataset : the over- and undersampled data (using SMOTE and ENN)
smote_enn = SMOTEENN(random_state=42)
X_smote_enn, y_smote_enn = smote_enn.fit_resample(X_train, y_train)

train_data_smote_enn = X_smote_enn.copy()
train_data_smote_enn["loan_status"] = y_smote_enn


# Visualize the datasets with different resampling techniques

y_oversampled = train_data_oversampled["loan_status"]
dataframes = [y_train, y_undersampled, y_oversampled, y_smote_enn]
titles = [
    "Original Dataset",
    "Under-sampled Dataset",
    "Over-sampled Dataset",
    "Over- and Under-sampled Dataset",
]

# Create the figure with subplots
fig, axes = plt.subplots(1, len(dataframes), figsize=(16, 5), sharey=True)
plt.subplots_adjust(wspace=0.3)
max_count = max(df.value_counts().max() for df in dataframes)

for i, (df, title) in enumerate(zip(dataframes, titles)):
    counts = df.value_counts().sort_index()
    total = counts.sum()
    percentages = (counts / total * 100).round(1)

    # Plot the barplot
    sns.barplot(x=counts.index, y=counts.values, ax=axes[i], palette="viridis")

    # Add percentages on top of the bars
    for j, val in enumerate(counts.values):
        axes[i].text(
            x=j,
            y=val + max_count * 0.03,
            s=f"{percentages.iloc[j]}%",
            ha="center",
            va="bottom",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

    axes[i].set_title(title)
    axes[i].set_xlabel("Loan Status")
    axes[i].set_xticks([0, 1])
    axes[i].set_xticklabels(["Default", "Fully Paid"])
    if i == 0:
        axes[i].set_ylabel("Count")
    axes[i].set_ylim(0, max_count * 1.2)
    axes[i].spines["top"].set_visible(False)
    axes[i].spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("figures/class_balances.png")


# =========================================================================================================================================
#                                                     FIT AND RUN THE MODELS
# =========================================================================================================================================

# Define the models

log_reg = LogisticRegression(max_iter=1000, random_state=42)

xgb_clf = XGBClassifier(
    use_label_encoder=False,
    n_estimators=50,
    tree_method="hist",
)

rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    max_features="log2",
    min_samples_split=50,
    min_samples_leaf=10,
    class_weight="balanced",
    bootstrap=True,
    n_jobs=-1,
    random_state=42,
)

knn_clf = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance",
    algorithm="auto",
)


def nn_model(num_columns, num_labels, hidden_units, dropout_rates, learning_rate):
    inp = tf.keras.layers.Input(shape=(num_columns,))
    x = BatchNormalization()(inp)
    x = Dropout(dropout_rates[0])(x)
    for i in range(len(hidden_units)):
        x = Dense(
            hidden_units[i],
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rates[i + 1])(x)
    x = Dense(num_labels, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(
        optimizer=Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[AUC(name="AUC")],
    )
    return model


num_columns = X_train.shape[1]
num_labels = 1
hidden_units = [200, 200, 200]
dropout_rates = [0.2, 0.2, 0.3, 0.2]
learning_rate = 1e-4
callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

ann_model = nn_model(
    num_columns=num_columns,
    num_labels=num_labels,
    hidden_units=hidden_units,
    dropout_rates=dropout_rates,
    learning_rate=learning_rate,
)


# Define the run function
def run_all_models(train_data, test_data, resampling_name):
    """
    This function runs all the models defined above to train them and make predictions on the test set
    It also saves the figure of the ROC curves in a .png file, and the performance results of the models in an .xlsx file

    Arguments :
        train_data (df) : this is the dataset the model is trained on (allowing to run the function with different resamplings)
        test_data (df) : this is the dataset the model is tested on to make predictions
        resampling_name (str) : name of the dataset to help differentiate the file names when saving them (figures, tables)

    Returns :
        list : the list of predictions, containing for each model the probabilities predicted, and the predictions with threshold 0.5
    """

    # Create a list to store predictions
    model_list = [log_reg, xgb_clf, rf_clf, knn_clf, ann_model]
    model_names = ["Logistic Regression", "XGBoost", "Random Forest", "KNN", "ANN"]
    pred_list = []
    for model in model_names:
        pred_list.append([model, pd.DataFrame(), pd.DataFrame()])

    # Split the X and y on the training and testing sets
    X_train = train_data.drop(columns=["loan_status"], axis=1)
    y_train = train_data["loan_status"]
    X_test = test_data.drop(columns=["loan_status"], axis=1)
    y_test = test_data["loan_status"]

    # Get rid of useless columns for training
    X_train = X_train.drop(columns=["id", "Unnamed: 0"], errors="ignore")
    X_test = X_test.drop(columns=["id", "Unnamed: 0"], errors="ignore")

    # Fit the models
    for i, model in enumerate(model_list[:-1]):  # [:-1] to not iterate it on ANN
        model.fit(X_train, y_train)

        # Predict probabilities
        pred_list[i][1] = model.predict_proba(X_test)[:, 1]

        # Predict classes (Threshold = 0.5)
        pred_list[i][2] = model.predict(X_test)

    # Fit ANN separately
    ann_model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=32,
        callbacks=[callback],
    )

    # Predict probabilities
    pred_list[-1][1] = ann_model.predict(X_test)

    # Predict classes (Threshold = 0.5)
    pred_list[-1][2] = (pred_list[-1][1] > 0.5).astype(int)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))

    for i, model in enumerate(pred_list):
        fpr, tpr, thresholds = roc_curve(y_test, model[1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model[0]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")

    plt.savefig("figures/roc_curve_" + resampling_name + ".png")

    # Create the table with the performance results
    results_list = []

    for model in pred_list:
        # Calculate metrics
        accuracy = accuracy_score(y_test, model[2])
        precision = precision_score(y_test, model[2], zero_division=0)
        recall = recall_score(y_test, model[2])
        f1 = f1_score(y_test, model[2])
        roc_auc = roc_auc_score(y_test, model[1])

        results_list.append([accuracy, precision, recall, f1, roc_auc])

    results_df = pd.DataFrame(
        results_list,
        columns=["Accuracy", "Precision", "Recall", "F1-score", "AUC"],
        index=[f"{model[0]}" for model in pred_list],
    )

    results_df.to_excel("tables/performance_" + resampling_name + ".xlsx")

    return pred_list


def confusion_matrices(pred_list, resampling_name):
    """
    This function generates and displays normalized confusion matrices from the models' predictions

    Arguments :
        pred_list (list) : list of lists returned by the run_all_models function
        resampling_name (str) : name of the dataset to help differentiate the file names when saving them
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 8))
    axes = axes.ravel()

    for i, model in enumerate(pred_list):
        conf_matrix = confusion_matrix(
            y_test,
            model[2],
            normalize="true",
        )

        sns.heatmap(
            conf_matrix,
            annot=True,
            cmap="Blues",
            ax=axes[i],
        )
        axes[i].set_title(f"Confusion Matrix of {model[0]}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    fig.delaxes(axes[5])

    plt.tight_layout()
    plt.savefig("figures/confusion_" + resampling_name + ".png")


pred_original = run_all_models(train_data_original, test_data, "original")
confusion_matrices(pred_original, "original")
pred_undersampled = run_all_models(train_data_undersampled, test_data, "undersampled")
pred_oversampled = run_all_models(train_data_oversampled, test_data, "oversampled")
pred_smote_enn = run_all_models(train_data_smote_enn, test_data, "smote_enn")
confusion_matrices(pred_smote_enn, "smote_enn")


# =========================================================================================================================================
#                                                     MONTE CARLO SIMULATION
# =========================================================================================================================================

# First let's visualize the importance of various features for XGBoost on SMOTE ENN resampling
booster = xgb_clf.get_booster()
importance = booster.get_score(importance_type="gain")
sorted_data = dict(
    sorted(importance.items(), key=lambda item: item[1], reverse=True)[:15]
)

names = list(sorted_data.keys())
values = list(sorted_data.values())

# Create the barplot
plt.figure(figsize=(10, 6))
bars = plt.barh(names, values, color="steelblue", height=0.3)

for bar, value in zip(bars, values):
    plt.text(
        bar.get_width() + 0.1,
        bar.get_y() + bar.get_height() / 2,
        f" {value:.2f}",
        va="center",
        fontsize=10,
    )
plt.xlim(0, 3000)
plt.gca().invert_yaxis()

plt.xlabel("Gain")
plt.tight_layout()
plt.savefig("figures/basis_feature_importance.png")


# Define the functions used for the Monte Carlo simulation
def generate_noise(data, noise_proportion, continuous_features, categories):
    # Gausssian noise
    noisy_data = data.copy()

    for column in continuous_features:
        std_column = noisy_data[column].std()
        mean_noise = 0
        std_noise = noise_proportion * std_column
        noisy_data[column] += np.random.normal(
            mean_noise, std_noise, noisy_data[column].shape
        )

    # Random Label noise
    category_like = pd.DataFrame()
    for category in categories:
        subset = data.filter(like=category, axis=1)
        category_like = pd.concat([category_like, subset], axis=1)

    for column in category_like.columns:
        random_flip = np.random.rand(noisy_data.shape[0]) < noise_proportion
        noisy_data.loc[random_flip, column] = ~noisy_data.loc[random_flip, column]

    return noisy_data


def monte_carlo_simulation(data, n_sim, noise_proportion, feature_list):
    data = data.drop(columns=["id", "Unnamed: 0"], errors="ignore")

    continuous_features = [
        "loan_amnt",
        "int_rate",
        "installment",
        "annual_inc",
        "dti",
        "open_acc",
        "pub_rec",
        "revol_bal",
        "revol_util",
        "total_acc",
        "mort_acc",
        "pub_rec_bankruptcies",
    ]

    categories = [
        "term",
        "sub_grade",
        "home_ownership",
        "verification_status",
        "purpose",
    ]

    importance_distrib = np.zeros((n_sim, len(feature_list)))
    for sim in range(n_sim):
        noisy_data = generate_noise(
            data=data,
            noise_proportion=noise_proportion,
            continuous_features=continuous_features,
            categories=categories,
        )
        xgb_clf.fit(noisy_data, y_train)

        # Get feature importance based on gain
        booster = xgb_clf.get_booster()
        importance = booster.get_score(importance_type="gain")
        for i, ft in enumerate(feature_list):
            importance_distrib[sim][i] = importance[ft]

    return importance_distrib


# Run the Monte Carlo simulation
feature_list = [
    "term_ 60 months",
    "int_rate",
    "verification_status_Source Verified",
    "home_ownership_RENT",
    "purpose_credit_card",
]

X_train = train_data_smote_enn.drop(columns=["loan_status"], axis=1)
y_train = train_data_smote_enn["loan_status"]
X_train = X_train.drop(columns=["id", "Unnamed: 0"], errors="ignore")

mc_distrib = monte_carlo_simulation(
    data=X_train, n_sim=500, noise_proportion=0.05, feature_list=feature_list
)

# Plot the distributions of the Gain for the different variables across the simulations
feature_list_labels = [
    "Verification Status",
    "Purpose : Credit Card",
    "Rented Home",
    "Interest Rate",
    "Term",
]
reordered_mc = mc_distrib[:, [2, 4, 3, 1, 0]]

plt.figure(figsize=(8, 6))

for i in range(reordered_mc.shape[1]):
    spread = max(reordered_mc[:, i]) - min(reordered_mc[:, i])
    bins = round(spread / 30)
    if bins <= 0:
        bins = 1
    sd = reordered_mc[:, i].std()
    plt.hist(
        reordered_mc[:, i],
        bins=bins,
        color=f"C{i}",
        alpha=0.7,
        edgecolor="black",
        label=f"{feature_list_labels[i]} (SD = {sd:.2f})",
    )

plt.title("Distributions of feature importance after simulation")
plt.xlabel("Gain")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()

plt.savefig("figures/monte_carlo_results.png")