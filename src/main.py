import hydra
import pandas as pd
from filter_text import filter_text
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split

vectorizer = TfidfVectorizer()


def load_data(file):
    df = pd.read_csv(file)

    df.drop_duplicates(inplace=True)
    return df


def save_data(file_name, df):
    df.to_csv(file_name, index=False)


@hydra.main(version_base=None, config_path="./../configs", config_name="config")
def train_model(
    cfg: DictConfig,
    train_path="./../csv/train.csv",
    test_size=0.2,
    c=4,
    max_iter=10000,
    class_weight='balanced',
    *args
):
    print(cfg.test.testval)
    train_df = load_data(train_path)

    train_df['filtered_text'] = train_df['comment_text'].apply(filter_text)

    filtered_texts_train = train_df['filtered_text'].values

    x_train_tfidf = vectorizer.fit_transform(filtered_texts_train)

    x = x_train_tfidf
    y = train_df.drop(['id', 'comment_text', 'filtered_text'], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    logistic_clf = MultiOutputClassifier(LogisticRegression(
        C=c,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=42, *args
    )
    )

    logistic_clf.fit(x_train, y_train)

    predictions = logistic_clf.predict(x_test)
    print(classification_report(y_test, predictions, target_names=y.columns))

    return x, y, logistic_clf


def predict(trained_model, y,
            test_path="./../csv/test.csv",
            result_path="./../csv/result.csv"
            ):
    test_df = load_data(test_path)

    test_df['filtered_text'] = test_df['comment_text'].apply(filter_text)

    filtered_texts_test = test_df['filtered_text'].values

    x_test_tfidf = vectorizer.transform(filtered_texts_test)

    test_predictions = trained_model.predict(x_test_tfidf)

    predictions_df = pd.DataFrame(test_predictions, columns=y.columns)

    predictions_df['id'] = test_df['id'].values

    predictions_df = predictions_df[['id'] + list(y.columns)]

    predictions_df.to_csv(result_path, index=False)


if __name__ == "__main__":
    x, y, model = train_model()
    predict(model, y)
