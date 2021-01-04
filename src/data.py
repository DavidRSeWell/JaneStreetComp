#import datatable as dt
import pandas as pd


class JaneData:

    def __init__(self,data_dir):
        self.example_test = self.load_df(data_dir + "/example_test.csv")
        self.features_df = self.load_df(data_dir + "/features.csv")
        self.train_df = self.load_df(data_dir + "/train.csv")

    @staticmethod
    def load_df(path: str) -> pd.DataFrame:
        return pd.read_csv(path)


