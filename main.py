import click
import json

from janestreet import utils

from janestreet.data import JaneData

@click.command()
@click.option("--config", help="Path to config file")
@click.option("--output_path",help="Path to save output")
def main(config,output_path):

    import pickle
    import time

    if type(config) == dict:
        print("Assuming passed in config is a correctly formated dictionary")
    else:
        config = utils.load_config(config)

    print("CONFIG")
    print("----------------------------------------------------------")
    print(json.dumps(config, indent=4, sort_keys=True))

    s_time = time.time()

    jane_data = JaneData(config["data_config"]["data_path"])

    train = jane_data.train_df

    train_features = jane_data.extract_features(train)

    steps, X = jane_data.transform_from_config(train_features.to_numpy(),config)

    y = train["resp"]

    pca , pca_config = steps["PCA"]

    kmeans,kmeans_config = steps["KMeans"]

    jane_data.display_histogram(steps["KMeans"][0].labels_, n_bins=kmeans_config["n_clusters"])

    def transform(x):
        features = [f"feature_{i}" for i in range(130)]
        x = x.fillna(0)
        x = x[features].to_numpy().reshape((1, len(features)))
        x = pca.transform(x)
        return x

    Agent , agent_config , Trainer = utils.load_agent(config["agent_config"])

    agent = object
    if Agent.__name__ == "SimpleKMeansBandit":
        agent = Agent(kmeans_config["n_clusters"], kmeans, transform,**agent_config)
    else:
        agent = Agent.load_from_config(config)

    exp_config = config["exp_config"]

    train_config = config.copy()

    try:
        del train_config["agent_config"]["agent"]
    except:
        pass

    train_config["agent"] = agent

    if Agent.__name__ == "SimpleKMeansBandit":
        train_config["data"] = train
    else:
        train_config["X"] = X
        train_config["y"] = y

    trainer = Trainer.load_from_config(train_config)

    trainer.train()

    if Agent.__name__ != "SimpleKMeansBandit":
        train_s , test_s = trainer.test_predict()

        print(train_s)
        print(test_s)

    if exp_config["submission"]:

        from datetime import datetime

        now = datetime.now()

        now = now.strftime("%H:%M:%S")

        pickle.dump(kmeans,open(output_path + f"/kmeans_{now}","wb"))

        pickle.dump(pca, open(output_path + "/pca", "wb"))

        agent.save(output_path)


    e_time = time.time()

    print()

    print("Submission file runtime")
    print((e_time - s_time) / 60.0)


if __name__ == "__main__":
    main()
