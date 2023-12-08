# Hate speech detector
This repository contains a machine learning classifier designed to identify toxicity and hate speech in Wikipedia comments. It is developed as part of a Kaggle competition.

## Directory Structure
- **eda**: Contains Jupyter notebooks for Exploratory Data Analysis (EDA).
- **configs**: Stores configuration files
- **src**:  Source code directory.
- **tests**: Contains test cases for the source code.
- **requirements.txt**: Lists all Python dependencies required to run the code.

## Installation

To work with this repository, you need to install the necessary dependencies:

```
pip install -r requirements.txt
```

## Testing

### Testing the Code

To test the code in the `src` directory:

```
pytest tests/code
```

### Testing the Datasets
For the training dataset:
```
pytest --dataset-loc=dataset/train.csv tests/data/test_traindataset.py --verbose --disable-warnings
```

For the test dataset:
```
pytest --dataset-loc=dataset/test.csv tests/data/test_testdataset.py --verbose --disable-warnings
```
To test a CSV that contains predictions:
```
pytest --dataset-loc=dataset/submission.csv tests/data/test_output.py --verbose --disable-warnings
```

## Code Style and Cleaning
To check and update the code style, use:

```make style``` - includes running black, flake8, isort, pyupgrade, and ruff.

To clean the repository:

``` make clean``` - will remove various temporary files and caches.


## Usage

### Training the Model

The code for training the model is in `src/main.py`. The configuration for training is in `configs/config.yaml`.

### Making Predictions

To make predictions for a given text using the trained model:

Example:
```
python src/inference.py --text="hello world"
```

