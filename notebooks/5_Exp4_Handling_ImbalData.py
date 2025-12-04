
# Libraries
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import emoji
import mlflow
import mlflow.sklearn
import dagshub


# MLFlow setup
dagshub.init(repo_owner='SirIsaac96', repo_name='youtube-feedback-sentiment-analysis', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/SirIsaac96/youtube-feedback-sentiment-analysis.mlflow')
mlflow.set_experiment('Exp 4 - Handling Imbalanced Data')


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


# Remove rows with empty cleaned_text
data = data[~(data['clean_text'].str.strip() == '')]


# Step 2: Feature Engineering
x = data['clean_text']
y = data['sentiment']


# Step 3: Function to run the experiment
def run_imbal_experiment(imbalanced_technique):
    # Feature extraction
    ngram_range = (1, 3) # Trigrams
    max_features = 1000
    vectorizer = CountVectorizer(ngram_range = ngram_range, max_features = max_features)
    
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = y)
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)

    # Handling class imbalance
    if imbalanced_technique == 'class_weights':
        class_weight = 'balanced'
    elif imbalanced_technique == 'oversampling':
        class_weight = None
        x_train, y_train = SMOTE(random_state=42).fit_resample(x_train, y_train)
    elif imbalanced_technique == 'undersampling':
        class_weight = None
        x_train, y_train = RandomUnderSampler(random_state=42).fit_resample(x_train, y_train)
    elif imbalanced_technique == 'adasyn':
        class_weight = None
        x_train, y_train = ADASYN(random_state=42).fit_resample(x_train, y_train)
    elif imbalanced_technique == 'smote_enn':
        class_weight = None
        x_train, y_train = SMOTEENN(random_state=42).fit_resample(x_train, y_train)

    # Model training with Random Forest
    with mlflow.start_run() as run:
        # Log a description for the run
        mlflow.set_tag('author', 'Isaac-Otom')
        mlflow.set_tag("model", "Exp4_HandlingImbalData")
        mlflow.set_tag('model_type', 'Random_Forest')
        mlflow.set_tag('description', f'Random Forest model for feedback sentiment analysis with BoW (1, 3) and max features = {max_features}, imbalance handling method = {imbalanced_technique}.')

        # Log parameters for the vectorizer
        mlflow.log_param('vectorizer_type', 'BoW')
        mlflow.log_param('ngram_range', ngram_range)
        mlflow.log_param('vectorizer_max_features', max_features)

        # Log Random Forest parameters
        n_estimators = 200
        max_depth = 15

        # Log parameters
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_param('max_depth', max_depth)
        mlflow.log_param('imbalance_handling', imbalanced_technique)

        # Train the model
        model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, 
                                       random_state = 42, class_weight = class_weight)
        model.fit(x_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(x_test)

        # Log metrics
        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric('accuracy', acc)

        # Classification report
        class_report = classification_report(y_test, y_pred, zero_division = 0, output_dict = True)
        for cls, metrics in class_report.items():
            if isinstance(metrics, dict):
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(f"{cls.replace(' ', '_')}_{metric_name}", metric_value)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize = (8, 6))
        sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues')
        plt.title(f'Confusion Matrix: BoW (1, 3), Max Features = {max_features}, Imbalance Handling = {imbalanced_technique}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        # Save and log the confusion matrix plot
        cm_path = f'conf_matrix_BoW_13_{imbalanced_technique}.png'
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()

        # Save and log the trained model
        model_path = f'Exp4_BoW_13_{imbalanced_technique}'
        mlflow.sklearn.save_model(model, model_path)
        mlflow.log_artifacts(model_path)

        # Log this script (5_Exp4_Handling_ImbalData.py) as an artifact
        try:
            mlflow.log_artifact(__file__)
        except Exception:
            pass


# Step 4: Run experiments for different imbalance handling techniques
imbalanced_techniques = ['class_weights', 'oversampling', 'undersampling', 'adasyn', 'smote_enn']


for technique in imbalanced_techniques:
    run_imbal_experiment(technique)
    

print('Training complete. Logged to MLFlow successfully.')
