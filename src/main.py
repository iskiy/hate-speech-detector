import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np

import filter_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier





@hydra.main(version_base=None, config_path="./../configs", config_name="config")
def train_model(train_path='./../dataset/train.csv', test_path='./../dataset/test.csv'):
    train_df = pd.read_csv(train_path)
    # test_df = pd.read_csv(test_path, usecols=['comment_text'])

    train_df['comment_text'] = train_df['comment_text'].apply(filter_text)

    X = train_df['comment_text'].values
    y = train_df.drop(['id', 'comment_text'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    forest_clf = MultiOutputClassifier(RandomForestClassifier())
    forest_clf.fit(X_train_tfidf, y_train)

    predictions = forest_clf.predict(X_test_tfidf)
    print(classification_report(y_test, predictions, target_names=y.columns))

    model = hydra.utils.instantiate(cfg.model)
    #
    # Train the model
    model.fit(X_train, y_train)
    #
    # Evaluate the model
    score = model.score(X_test, y_test)
    print(f"Model Score: {score}")
#

if __name__ == "__main__":
    train_model()
