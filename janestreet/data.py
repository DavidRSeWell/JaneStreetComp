import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import load_sklearn

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class JaneData:

    @classmethod
    def load_from_config(cls,config):
        data_dir = config["data_config"]["data_path"]

    def __init__(self,data_dir,seed=None):
        self.example_test = self.load_df(data_dir + "/example_test.csv")
        self.features_df = self.load_df(data_dir + "/features.csv")
        self.train_df = self.load_df(data_dir + "/train.csv")

        self._pca = None
        self._seed = seed if (seed is not None) else np.random.seed(np.random.randint(1000))

        print("Done Loading Jane Data")
        print(f"Using seed data {self._seed}")

    def clean_data(self,data: pd.DataFrame) -> pd.DataFrame:

        #data = data.apply(lambda x: self.replace_with_mean(x))
        data = data.fillna(0)

        return data

    def extract_features(self,data) -> pd.DataFrame:
        feature_col = [f"feature_{i}" for i in range(130)]

        return self.clean_data(data[feature_col])

    def create_transformed(self,pca, features, n_top_components):
        ''' Return a dataframe of data points with component features.
            The dataframe should be indexed by State-County and contain component values.
            :param train_pca: A list of pca training data, returned by a PCA model.
            :param features: A dataframe of features.
            :param n_top_components: An integer, the number of top components to use.
            :return: A dataframe with n_top_component values as columns.
         '''
        # create new dataframe to add data to
        transformed = pca.transform(features)

        # keep only the top n components
        transformed = transformed[:, :n_top_components]

        return transformed

    def display_pca_components(self,components,features,min_idx,n_weights):

        # get the list of weights from a row in v, dataframe
        v_1 = components[min_idx, :]

        # match weights to features in counties_scaled dataframe, using list comporehension
        comps = pd.DataFrame(list(zip(v_1, features)),
                             columns=['weights', 'features'])

        # we'll want to sort by the largest n_weights
        # weights can be neg/pos and we'll sort by magnitude
        comps['abs_weights'] = comps['weights'].apply(lambda x: np.abs(x))
        sorted_weight_data = comps.sort_values('abs_weights', ascending=False).head(n_weights)

        # display using seaborn
        ax = plt.subplots(figsize=(10, 6))
        ax = sns.barplot(data=sorted_weight_data,
                         x="weights",
                         y="features",
                         palette="Blues_d")
        ax.set_title("PCA Component Makeup, Component #" + str(min_idx))
        plt.show()

    def display_histograms(self,cols,n_bins=50):
        """
        Display a group of colums as histograms
        :param cols:
        :return:
        """
        for column_name in cols:
            self.display_histogram(self.train_df[column_name],n_bins)

    def display_histogram(self,data,n_bins=0):

        K = len(set(data)) if (n_bins == 0) else n_bins
        ax = plt.subplots(figsize=(6, 3))
        ax = plt.hist(data, bins=K, color='blue', rwidth=0.5)
        title = "Histogram of Cluster Counts"
        plt.title(title, fontsize=12)
        plt.show()

    @staticmethod
    def transform_from_config(X: np.ndarray, config: dict) -> (dict,np.ndarray):

        data_config = config["data_config"]

        steps = {}
        X = X.copy()
        for step in data_config["steps"].keys():
            config = data_config["steps"][step]
            sklearn_trans , name = load_sklearn(step,config)
            if "cluster" in name:
                cluster = sklearn_trans.fit(X)
                #TODO Not all sklearn cluster models have labels_ attribute
                steps[step] = (cluster,config)
            elif "decomposition" in name:
                X = sklearn_trans.fit_transform(X)
                steps[step] = (sklearn_trans,config)

        return steps,X

    @staticmethod
    def run_pca(data,n):
        print("Running PCA")
        print(f"Data shape {data.shape}")

        data_np = data.to_numpy()
        pca = PCA(n_components=n)
        pca.fit(data_np)

        return pca

    @staticmethod
    def run_kmean(n,X):
        kmeans = KMeans(n_clusters=n, random_state=0).fit(X)
        return kmeans

    @staticmethod
    def get_min_num(pca, min_exp=0.80):
        """
        Returns number of principle components that are
        necessary to reach a given min explained variance
        threshold
        """
        explained_var = pca.explained_variance_ratio_.cumsum()

        return np.where(explained_var >= min_exp)[0][0] - 1

    @staticmethod
    def load_df(path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    @staticmethod
    def replace_with_mean(x):
        mean = x.mean()
        x[x.isna()] = mean
        return x





