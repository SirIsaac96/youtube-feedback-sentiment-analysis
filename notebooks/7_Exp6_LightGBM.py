
# Necessary libraries
import re
import pandas as pd
import numpy as np
import optuna
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
import emoji
import dagshub
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# MLflow / DagsHub setup
dagshub.init(
    repo_owner="SirIsaac96",
    repo_name="youtube-feedback-sentiment-analysis",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/SirIsaac96/youtube-feedback-sentiment-analysis.mlflow"
)
mlflow.set_experiment("Exp 6 - LightGBM with detailed Hyperparameter Tuning")


# Step 1: Load and preprocess data
data = pd.read_csv("data_raw/youtube_sentiments.csv")
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)


# Text preprocessing function
def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Remove mentions (@user123)
    text = re.sub(r'@\w+', '', text)

    # 3. Remove newline characters
    text = re.sub(r'\n', ' ', text)

    # 4. Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)

    # 5. Remove punctuations, numbers, and special chars except hashtags
    text = re.sub(r'[^a-zA-Z#\s]', '', text)

    # 6. Emoji handling
    text = emoji.demojize(text, delimiters=(' ', ' '))

    # 7. Remove stopwords
    stop_words = set(stopwords.words('english')) - {'not', 'no', 'but', 'however', 'yet', 'you'}
    text = ' '.join([w for w in text.split() if w not in stop_words])

    # 8. Remove short words (less than 3 characters)
    important_short = {"not", "bad", "yes", "wow", "fun", "win"}
    text = ' '.join([w for w in text.split() if len(w) >= 3 or w in important_short])

    # 9. Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in text.split()]

    return ' '.join(tokens)


data["clean_text"] = data["text"].apply(preprocess_text)


# Remap the class labels from [-1, 0, 1] to [2, 0, 1]
data["sentiment"] = data["sentiment"].map({-1: 2, 0: 0, 1: 1})


# Remove any rows with empty cleaned text
data = data[data["clean_text"].str.strip() != ""]


# Step 2: BoW Vectorization
vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=1000)
x_bow = vectorizer.fit_transform(data["clean_text"]).astype(np.float32)
y = data["sentiment"]


# Step 3: Apply SMOTE to handly class imbalance
x_resampled, y_resampled = SMOTE(random_state=42).fit_resample(x_bow, y)


# Step 4: Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42,
    stratify=y_resampled
)


# Step 5: MLflow logging function
def log_mlflow_results(model, x_train, x_test, y_train, y_test, params, trial_number):
    with mlflow.start_run(nested=True):
        # Log a description for the run
        mlflow.set_tag("author", "Isaac-Otom")
        mlflow.set_tag("model_type", "LightGBM Classifier")
        mlflow.set_tag(
            "mlflow.runName",
            f"Trial_{trial_number}_LightGBM_BoW_SMOTE"
        )

        # Log hyperparameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        # Fit the model
        model.fit(x_train, y_train)

        # Predictions
        y_pred = model.predict(x_test)

        # Log metrics
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        for cls, metrics in report.items():
            if isinstance(metrics, dict):
                for m_name, m_val in metrics.items():
                    mlflow.log_metric(f"{cls}_{m_name}", m_val)

        # Log the model
        mlflow.sklearn.log_model(
            model,
            name=f"trial_{trial_number}_model"
        )

        return accuracy


# Step 6: Hyperparameter Tuning for LightGBM Classifier (Optuna objective function)
def lightgbm_objective(trial):
    # Define the hyperparameter search space
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 128),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 40),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.2),
        "random_state": 42
    }

    # Create the LightGBM model and log results to MLflow
    model = LGBMClassifier(**params)
    return log_mlflow_results(model, x_train, x_test, y_train, y_test, params, trial.number)


# Step 7: Run Optuna + log BEST model once
# =========================
def run_optuna_study():
    with mlflow.start_run(run_name="Optuna_LightGBM_Study"):
        study = optuna.create_study(direction="maximize")
        study.optimize(lightgbm_objective, n_trials=15)

        # Get the best trial and log the best model to MLFlow
        best_params = study.best_params
        best_params.update({
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "random_state": 42
        })

        best_model = LGBMClassifier(**best_params)
        best_model.fit(x_train, y_train)

        mlflow.log_params(best_params)
        mlflow.log_metric("best_accuracy", study.best_value)

        mlflow.sklearn.log_model(
            best_model,
            name="best_model"
        )

        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image("lightgbm_param_importance.png")
        mlflow.log_artifact("lightgbm_param_importance.png")


# Step 8: Execute the Optuna study
run_optuna_study()
