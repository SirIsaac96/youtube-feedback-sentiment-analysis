
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
from xgboost import XGBClassifier
import emoji
import dagshub
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# MLFlow setup
dagshub.init(repo_owner='SirIsaac96', repo_name='youtube-feedback-sentiment-analysis', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/SirIsaac96/youtube-feedback-sentiment-analysis.mlflow')
mlflow.set_experiment('Exp 5 - XGBoost with Hyperparameter Tuning')


# Step 1: Load and preprocess the data
data = pd.read_csv('data_raw/youtube_sentiments.csv')


# Drop missing and duplicate values
data.dropna(inplace = True)
data.drop_duplicates(inplace = True)


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

    # 9. Tokenization
    tokens = text.split()

    # 10. Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return ' '.join(tokens)


data['clean_text'] = data['text'].apply(preprocess_text)


# Remap the class labels from [-1, 0, 1] to [2, 0, 1]
data['sentiment'] = data['sentiment'].map({-1: 2, 0: 0, 1: 1})


# Remove rows with empty cleaned_text
data = data[~(data['clean_text'].str.strip() == '')]


# Step 2: Feature Engineering
x = data['clean_text']
y = data['sentiment']


# Step 3: Feature Extraction using BoW with n-grams (1 to 3)
ngram_range = (1, 3) # Trigrams
max_features = 1000


# Train-test split before vectorization and resampling
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = y)


# Vectorization using BoW
vectorizer = CountVectorizer(ngram_range = ngram_range, max_features = max_features)
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)


# Resampling using SMOTE
x_train_vec, y_train = SMOTE(random_state = 42).fit_resample(x_train_vec, y_train)


# Step 4: Log results to MLFlow
def log_mlflow_results(model_name, model, x_train, x_test, y_train, y_test):
    with mlflow.start_run():
        # Log a description for the run
        mlflow.set_tag('author', 'Isaac-Otom')
        mlflow.set_tag("model", "Exp5_XGBoostwithHPTuning")
        mlflow.set_tag('model_type', 'XGBoost Classifier')
        mlflow.set_tag('description', f'XGBoost Classifier for feedback sentiment analysis with Hyperparameter Tuning using SMOTE resampling.')

        # Log algorithm name as a parameter
        mlflow.log_param('algorithm', model_name)

        # Fit the model
        model.fit(x_train, y_train)
        
        # Predictions
        y_pred = model.predict(x_test)

        # Log metrics
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric('accuracy', accuracy)

        # Classification report
        class_report = classification_report(y_test, y_pred, zero_division = 0, output_dict = True)
        for cls, metrics in class_report.items():
            if isinstance(metrics, dict):
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(f"{cls.replace(' ', '_')}_{metric_name}", metric_value)

        # Save and log the model
        model_path = f'Exp5_{model_name}'
        mlflow.sklearn.save_model(model, model_path)
        mlflow.log_artifacts(model_path)


# Step 5: Hyperparameter Tuning for XGBoost Classifier (Optuna objective function)
def xgboost_objective(trial):
    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss'
    }

    model = XGBClassifier(**params)
    return accuracy_score(y_test, model.fit(x_train_vec, y_train).predict(x_test_vec))


# Step 6: Run Optuna study for hyperparameter tuning, log only best model to MLFlow
def run_optuna_study():
    study = optuna.create_study(direction = 'maximize')
    study.optimize(xgboost_objective, n_trials = 15)

    # Get the best trial and onlog the best model to MLFlow
    best_params = study.best_params
    best_params.update({
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        'random_state': 42})
    best_model = XGBClassifier(**best_params)

    # Log the best model to MLFlow, passing the algorithm name as "XGBoost_Classifier"
    log_mlflow_results("XGBoost_Classifier", best_model, x_train_vec, x_test_vec, y_train, y_test)


# Run the Optuna study
run_optuna_study()
