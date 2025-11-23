
# Libraries
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
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
mlflow.set_experiment('Exp 1 - Baseline Moldel (Random Forest)')
# mlflow.set_tracking_uri('http://127.0.0.1:5000')


# Step 1: Load and preprocess the data
data = pd.read_csv('data_raw/youtube_sentiments.csv')


# Drop missing and duplicate values
data.dropna(inplace = True)
data.drop_duplicates(inplace = True)


# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')


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


# Step 2: Feature extraction using Bag of Words
vectorizer = CountVectorizer(max_features = 10000) # Limit to top 10,000 features
x = vectorizer.fit_transform(data['clean_text'])
y = data['sentiment']


# Step 3: Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = y)


# Step 4: Model training with Random Forest
with mlflow.start_run() as run:
    # Log a description for the run
    mlflow.set_tag('author', 'Isaac-Otom')
    mlflow.set_tag("model", "Exp1_Baseline")
    mlflow.set_tag('model_type', 'Random_Forest')
    mlflow.set_tag('description', 'Baseline Random Forest model for feedback sentiment analysis using Bag of Words features')

    # Log parameters for the vectorizer
    mlflow.log_param('vectorizer_type', 'CountVectorizer')
    mlflow.log_param('vectorizer_max_features', vectorizer.max_features)

    # Log Random Forest parameters
    n_estimators = 500
    max_depth = 15

    # Log parameters
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('max_depth', max_depth)

    # Train the model
    model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, random_state = 42)
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
            for metric, value in metrics.items():
                mlflow.log_metric(f"{cls}_{metric}", value)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize = (8, 6))
    sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save and log the confusion matrix plot
    plt.savefig('conf_matrix_exp1.png')
    mlflow.log_artifact('conf_matrix_exp1.png')

    # Save and log the trained model
    mlflow.sklearn.save_model(sk_model = model, path = 'Exp1RandomForestBaseline')
    mlflow.log_artifacts('Exp1RandomForestBaseline')

    # Log this script (2_Exp1_BaselineModel.py) as an artifact
    try:
        mlflow.log_artifact(__file__)
    except Exception:
        pass


print('Training complete. Logged to Dagshub MLFlow successfully.')
