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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc)
from sklearn.naive_bayes import GaussianNB

########## 2. Define reusable preprocessing ##########

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
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
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

########## 3. Define main function per project ##########

def run_project(project, repeat=20):
    print(f"\n▶ Running project: {project}")
    path = f'datasets/{project}.csv'
    
    # Load and shuffle data
    pd_all = pd.read_csv(path)
    pd_all = pd_all.sample(frac=1, random_state=999)

    pd_all['Title+Body'] = pd_all.apply(
        lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
        axis=1
    )

    pd_tplusb = pd_all.rename(columns={
        "Unnamed: 0": "id",
        "class": "sentiment",
        "Title+Body": "text"
    })

    data = pd_tplusb[["id", "Number", "sentiment", "text"]].fillna('')
    text_col = 'text'

    # Preprocess
    data[text_col] = data[text_col].apply(preprocess_text)
    data['text_length'] = data[text_col].apply(lambda x: len(x.split()))
    data['perf_keyword_count'] = data[text_col].apply(keyword_counter)

    params = {'var_smoothing': np.logspace(-12, 0, 13)}

    accuracies, precisions, recalls, f1_scores, auc_values = [], [], [], [], []

    for i in range(repeat):
        indices = np.arange(data.shape[0])
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=i)

        train_text = data[text_col].iloc[train_idx]
        test_text = data[text_col].iloc[test_idx]
        y_train = data['sentiment'].iloc[train_idx]
        y_test = data['sentiment'].iloc[test_idx]

        tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
        X_train_tfidf = tfidf.fit_transform(train_text).toarray()
        X_test_tfidf = tfidf.transform(test_text).toarray()

        train_metadata = data[['text_length', 'perf_keyword_count']].iloc[train_idx].to_numpy()
        test_metadata = data[['text_length', 'perf_keyword_count']].iloc[test_idx].to_numpy()

        X_train = np.hstack((X_train_tfidf, train_metadata))
        X_test = np.hstack((X_test_tfidf, test_metadata))

        clf = GaussianNB()
        grid = GridSearchCV(clf, params, cv=5, scoring='roc_auc')
        grid.fit(X_train, y_train)
        best_clf = grid.best_estimator_
        best_clf.fit(X_train, y_train)
        y_pred = best_clf.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average='macro'))
        recalls.append(recall_score(y_test, y_pred, average='macro'))
        f1_scores.append(f1_score(y_test, y_pred, average='macro'))

        fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
        auc_values.append(auc(fpr, tpr))

    # Aggregate
    result = {
        'project': project,
        'repeated_times': repeat,
        'Accuracy': np.mean(accuracies),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'F1': np.mean(f1_scores),
        'AUC': np.mean(auc_values),
        'CV_list(AUC)': str(auc_values)
    }

    # Save results
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{project}_NB.csv")

    df_log = pd.DataFrame([result])
    if not os.path.exists(out_path):
        df_log.to_csv(out_path, index=False)
    else:
        df_log.to_csv(out_path, mode='a', header=False, index=False)

    print(f" Saved results to {out_path}")
    return result

########## 4. Run all projects ##########

if __name__ == "__main__":
    all_projects = ['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe']
    all_results = []

    for proj in all_projects:
        result = run_project(proj)
        all_results.append(result)

    # Show summary
    print("\n Summary of All Projects:")
    summary_df = pd.DataFrame(all_results)
    print(summary_df[['project', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']])
