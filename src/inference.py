from joblib import load
import requests
from io import BytesIO
from filter_text import filter_text
import click

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def download_and_load_file(url):
    response = requests.get(url)
    response.raise_for_status()
    model_file = BytesIO(response.content)
    model = load(model_file)
    return model


def predict(text, model, vectorizer):
    filtered_text = filter_text(text)
    vectorizer.transform(filtered_text)
    return model.predict()


@click.command()
@click.option('--model_url',
              default='https://github.com/your-repo/your-model-path/model.joblib',
              prompt='Model URL', help='URL to download the model')
@click.option('--vectorizer_url',
              default='https://github.com/iskiy/hate-speech-detector/tree/main/configs/tfidf_vectorizer.joblib',
              prompt='vectorizer URL', help='URL to the TF-IDF vectorizer')
@click.option('--text', default=None, prompt='Enter text to classify', help='Text to classify')
def main(model_url, vectorizer_url, text):
    model = download_and_load_file(model_url)
    vectorizer = download_and_load_file(vectorizer_url)

    prediction_array = predict(text, model, vectorizer)[0]
    predictions = []
    for i, class_label in enumerate(class_names):
        if prediction_array[i] == 1:
            predictions.append(class_label)

    print(predictions)


if __name__ == "__main__":
    main()
