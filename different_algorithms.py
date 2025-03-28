########## 1. Import required libraries ##########
import pandas as pd
import numpy as np
import re
import os
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from scipy.sparse import hstack

# === Preprocessing ===
custom_stop_words_list = ['model', 'training', 'run', 'data', 'use', 'network', 'batch', 'gpu']
final_stop_words_list = set(stopwords.words('english') + custom_stop_words_list)
lemmatizer = WordNetLemmatizer()

def remove_html(text):
    return re.sub(r'<.*?>', '', text)

def remove_emoji(text):
    emoji_pattern = re.compile("[" 
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", text)
    text = re.sub(r"\'s", " 's", text)
    text = re.sub(r"\'ve", " 've", text)
    text = re.sub(r"\)", " ) ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"[\\'\"]", "", text)
    return text.strip().lower()

def preprocess_text(text):
    text = remove_html(text)
    text = remove_emoji(text)
    text = clean_str(text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in final_stop_words_list]
    return " ".join(tokens)

def keyword_counter(text):
    perf_keywords = ['slow', 'latency', 'speed', 'delay', 'performance', 'fast', 'optimize', 'optimization']
    return sum(word in text.lower() for word in perf_keywords)

# === Classifier Mapping ===
MODELS = {
    "GaussianNB": (GaussianNB(), False),
    "MultinomialNB": (MultinomialNB(), True),
    "LogisticRegression": (LogisticRegression(max_iter=1000), True),
    "LinearSVM": (LinearSVC(), True),
    "RandomForest": (RandomForestClassifier(n_estimators=100), True)
}

PROJECTS = ['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe']
REPEAT = 20

results = []

for model_name, (clf, supports_sparse) in MODELS.items():
    print(f"\n========== MODEL: {model_name} ==========")
    for project in PROJECTS:
        print(f"\n▶ Project: {project}")
        path = f'datasets/{project}.csv'

        df = pd.read_csv(path).sample(frac=1, random_state=999)
        df['Title+Body'] = df.apply(lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'], axis=1)
        df = df.rename(columns={"Unnamed: 0": "id", "class": "sentiment", "Title+Body": "text"})
        data = df[["id", "Number", "sentiment", "text"]].fillna('')

        data['text'] = data['text'].apply(preprocess_text)
        data['text_length'] = data['text'].apply(lambda x: len(x.split()))
        data['perf_keyword_count'] = data['text'].apply(keyword_counter)

        text_col = 'text'
        accuracy_list, precision_list, recall_list, f1_list, auc_list = [], [], [], [], []

        for i in range(REPEAT):
            train_idx, test_idx = train_test_split(np.arange(data.shape[0]), test_size=0.2, random_state=i)
            train_text = data[text_col].iloc[train_idx]
            test_text = data[text_col].iloc[test_idx]
            y_train = data['sentiment'].iloc[train_idx]
            y_test = data['sentiment'].iloc[test_idx]

            tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=1000)
            X_train_tfidf = tfidf.fit_transform(train_text)
            X_test_tfidf = tfidf.transform(test_text)

            train_meta = data[['text_length', 'perf_keyword_count']].iloc[train_idx].to_numpy()
            test_meta = data[['text_length', 'perf_keyword_count']].iloc[test_idx].to_numpy()

            if supports_sparse:
                X_train = hstack([X_train_tfidf, train_meta])
                X_test = hstack([X_test_tfidf, test_meta])
            else:
                X_train = np.hstack([X_train_tfidf.toarray(), train_meta])
                X_test = np.hstack([X_test_tfidf.toarray(), test_meta])

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            accuracy_list.append(accuracy_score(y_test, y_pred))
            precision_list.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
            recall_list.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
            f1_list.append(f1_score(y_test, y_pred, average='macro', zero_division=0))

            fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
            auc_list.append(auc(fpr, tpr))

        results.append({
            'Model': model_name,
            'Project': project,
            'Accuracy': np.mean(accuracy_list),
            'Precision': np.mean(precision_list),
            'Recall': np.mean(recall_list),
            'F1': np.mean(f1_list),
            'AUC': np.mean(auc_list)
        })

# Save summary
results_df = pd.DataFrame(results)
os.makedirs("outputs", exist_ok=True)
results_df.to_csv("outputs/all_models_results.csv", index=False)
print("\n All results saved to outputs/all_models_results.csv")
