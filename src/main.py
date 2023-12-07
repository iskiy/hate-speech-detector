import hydra
# import numpy as np
# import pandas as pd
from omegaconf import DictConfig

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import classification_report
# from sklearn.model_selection import train_test_split
# from sklearn.multioutput import MultiOutputClassifier
# import filter_text


@hydra.main(version_base=None, config_path="./../configs", config_name="config")
def train_model(
    cfg: DictConfig,
    train_path="./../dataset/train.csv",
    test_path="./../dataset/test.csv",
):
    print(cfg.test.testval)


if __name__ == "__main__":
    train_model()
