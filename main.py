import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from src.bandits import SimpleKMeansBandit
from src.trainer import BanditTrainer, TrainEnv

#import janestreet
from src.data import JaneData

def run(data_dir,K=50,eps=0.1,top_n=10):

    jane_data = JaneData(data_dir)

    n = len(jane_data.train_df)

    n_train = int(n*0.9)

    train , test = jane_data.train_df.iloc[:n_train,:] , jane_data.train_df.iloc[n_train:,:]

    train_env = TrainEnv(train)

    test_env = TrainEnv(test)

    train_features = jane_data.extract_features(train)

    pca = jane_data.run_pca(train_features,top_n)

    X = jane_data.create_transformed(pca,train_features,top_n)

    kmeans = jane_data.run_kmean(K,X)

    jane_data.display_histogram(kmeans.labels_,n_bins=K)

    def transform(x):
        features = [f"feature_{i}" for i in range(130)]
        x = x.fillna(0)
        x = x[features].to_numpy().reshape((1,len(features)))
        x = pca.transform(x)
        return x

    agent = SimpleKMeansBandit(K,kmeans,transform,threshold=0,eps=eps)

    trainer = BanditTrainer(agent,test_env,train_env)

    train_r , test_r = trainer.train(iters=50)
    x_axis = [x for x in range(len(train_r))]
    plt.plot(x_axis,train_r,label="train")
    plt.legend()
    plt.show()

    plt.plot(x_axis, test_r, label="test")
    plt.legend()
    plt.show()

def submission_run(data_dir,K=50,eps=0.1,top_n=10,save_path=""):

    import pickle
    import time

    s_time = time.time()

    jane_data = JaneData(data_dir)

    train = jane_data.train_df

    train_env = TrainEnv(train)

    train_features = jane_data.extract_features(train)

    pca = jane_data.run_pca(train_features, top_n)

    X = jane_data.create_transformed(pca, train_features, top_n)

    kmeans = jane_data.run_kmean(K, X)

    jane_data.display_histogram(kmeans.labels_, n_bins=K)

    def transform(x):
        features = [f"feature_{i}" for i in range(130)]
        x = x.fillna(0)
        x = x[features].to_numpy().reshape((1, len(features)))
        x = pca.transform(x)
        return x

    agent = SimpleKMeansBandit(K, kmeans, transform, threshold=0, eps=eps)

    trainer = BanditTrainer(agent, train_env)

    train_r, test_r = trainer.train(iters=1)

    class SubmitAgent:
        def __init__(self,kmeans,pca,Q):
            self.kmeans = kmeans
            self.pca = pca
            self.Q = Q

        def predict(self,x):

            features = [f"feature_{i}" for i in range(130)]
            x = x.fillna(0)
            x = x[features].to_numpy().reshape((1, len(features)))
            x = pca.transform(x)
            arm = self.kmeans.predict(x)[0]
            q_val = self.Q[arm]
            if q_val > 0: return 1
            else: return 0

    submit_agent = SubmitAgent(kmeans,pca,agent.Q)

    pickle.dump(kmeans,open(data_dir + "/kmeans","wb"))
    pickle.dump(pca,open(data_dir + "/pca","wb"))
    pickle.dump(list(agent.Q),open(data_dir + "/lookup","wb"))

    e_time = time.time()

    print("Submission file runtime")
    print((e_time - s_time) / 60.0)

    #pickle.dump(submit_agent,data_dir + "/submit_agent")





if __name__ == "__main__":

    st = time.time()
    data_dir = "/Users/davidsewell/MLData/JaneStreet"
    #train_df = pd.read_csv(data_dir + "/train.csv")
    #train_df = train_df[10000:110000]
    #train_df.to_csv(data_dir + "/train_small_2.csv")
    #run(data_dir)
    submission_run(data_dir)
    et = time.time()
    rt = (et - st) / 60.0
    print(f"Runtime {rt}")
