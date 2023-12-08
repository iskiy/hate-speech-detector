import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

from filter_text import filter_text


def load_data(file):
    df = pd.read_csv(file)

    df.drop_duplicates(inplace=True)
    return df


def save_data(file_name, df):
    df.to_csv(file_name, index=False)


@hydra.main(version_base=None, config_path="./../configs", config_name="config")
def train_model(
    cfg: DictConfig
):
    vectorizer = TfidfVectorizer(**cfg.vectorizer_params)
    train_df = load_data(cfg.data.train_path)

    train_df["filtered_text"] = train_df["comment_text"].apply(filter_text)
    filtered_texts_train = train_df["filtered_text"].values

    x_train_tfidf = vectorizer.fit_transform(filtered_texts_train)

    x = x_train_tfidf
    y = train_df.drop(["id", "comment_text", "filtered_text"], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=cfg.test_size, random_state=42
    )

    logistic_clf = MultiOutputClassifier(LogisticRegression(**cfg.model_params, random_state=42))
    logistic_clf.fit(x_train, y_train)

    predictions = logistic_clf.predict(x_test)
    print(classification_report(y_test, predictions, target_names=y.columns))

    return y, logistic_clf, vectorizer

def predict(
    trained_model, y, vectorizer, cfg: DictConfig
):
    test_df = load_data(cfg.data.test_path)

    test_df["filtered_text"] = test_df["comment_text"].apply(filter_text)
    filtered_texts_test = test_df["filtered_text"].values

    x_test_tfidf = vectorizer.transform(filtered_texts_test)

    test_predictions = trained_model.predict(x_test_tfidf)
    predictions_df = pd.DataFrame(test_predictions, columns=y.columns)
    predictions_df["id"] = test_df["id"].values
    predictions_df = predictions_df[["id"] + list(y.columns)]

    save_data(cfg.data.result_path, predictions_df)


if __name__ == "__main__":
    y, model, vectorizer = train_model()
    cfg = hydra.core.hydra_config.HydraConfig.get()
    predict(model, y, vectorizer, cfg)
