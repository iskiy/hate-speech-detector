import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

from filter_text import filter_text


def load_data(file):
    df = pd.read_csv(file)

    df.drop_duplicates(inplace=True)
    return df


def save_data(file_name, df):
    df.to_csv(file_name, index=False)


def train_model(
        cfg: DictConfig
):
    tfidf = TfidfVectorizer(**cfg.vectorizer_params)

    train_df = load_data(cfg.data.train_path)
    train_df["filtered_text"] = train_df["comment_text"].apply(filter_text)
    filtered_texts_train = train_df["filtered_text"].values

    x_train_tfidf = tfidf.fit_transform(filtered_texts_train)

    x = x_train_tfidf
    y = train_df.drop(["id", "comment_text", "filtered_text"], axis=1)

    logistic_clf = MultiOutputClassifier(LogisticRegression(**cfg.model_params, random_state=42))
    logistic_clf.fit(x, y)

    return logistic_clf, y, tfidf


def predict(
        trained_model, y, tfidf, cfg: DictConfig
):
    test_df = load_data(cfg.data.test_path)

    test_df["filtered_text"] = test_df["comment_text"].apply(filter_text)
    filtered_texts_test = test_df["filtered_text"].values

    x_test_tfidf = tfidf.transform(filtered_texts_test)

    test_predictions = trained_model.predict_proba(x_test_tfidf)
    predictions_df = pd.DataFrame(columns=y.columns)
    predictions_df["id"] = test_df["id"].values
    predictions_df = predictions_df[["id"] + list(y.columns)]

    for i, col in enumerate(predictions_df.columns[1:]):
        predictions_df[col] = test_predictions[i - 1][0, 1]

    save_data(cfg.data.result_path, predictions_df)


@hydra.main(version_base=None, config_path="./../configs", config_name="config")
def main(
        cfg: DictConfig
):
    model, y, tfidf = train_model(cfg)
    predict(model, y, tfidf, cfg)


if __name__ == "__main__":
    main()
