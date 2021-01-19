# JaneStreetComp

Repository for hosting my solution to the Jane Street Kaggle Competition. The goal of this repo was to test a type of multi arm bandit agent and see if it could do better than a simple base line logistic regression model. The short answer is that using a simple bandit algorithm could not beat a simple baseline but was nevertheless an intersting excercise. 

# Download Jane Street data

Go to https://www.kaggle.com/c/jane-street-market-prediction/data and download the data

# Running a model

1. Clone repo
```
git clone https://github.com/befeltingu/JaneStreetComp.git
cd JaneStreetComp
```

2. Install dependencies
As always perferably in a virtual environment

```
pip install -r requirements.txt
```

3. Run a model

```
python main.py --config=/path_to_config --output_path=path_to_save_model
```

The outpu_path variable describes the path where you want to save your model output

### Defining a config

In the *etc/* directory there are two example configs that you can use or change for yourself
There are currently two models to run a logistic regression based agent that you can run using the etc/sl_config.yaml
The other is a Multi arm bandit agent runnable through etc/bandit_config.yaml

4. Push saved model to Kaggle and submit. 

In order to use the saved output on kaggle you will have to upload all parts (kmeans,pca and or sklearn LR model) as a dataset on kaggle. 

For example if I ran a bandit algorithm it will output 3 pieces. Kmeans, pca and a lookup. The lookup is a dictionary set of q values. I upload those three pieces to kaggle by clicking on the *Data* button on the left sidebar of the kaggle dashboard if your already logged in. The there is a *New DataSet* button that you can click to create new dataset. Then you will upload those 3 output pieces from your local computer. Now that you have uploaded the pieces they are available through any Jane Street notebook. Below is how you might go about defining an agent in a notebook. To submit to kaggle you have to run a notebook and then save it by click *save version* in the top right corner. Then go back to the Jane Street Challenge page and click make submission. Your notebook version should be available to submit. Then wait for your score to appear on the leaderboard. 

Define agent class. Notice the paths to the 3 pieces. 
```
class SubmitClass:
    @classmethod
    def load_from_disk(cls,data_dir):
        pca = pickle.load(open(data_dir + "/pca","rb"))
        lookup = pickle.load(open(data_dir + "/lookup_192734","rb"))
        kmeans = pickle.load(open(data_dir + "/kmeans_192734","rb"))
        return cls(kmeans,lookup)

    def __init__(self,kmeans,lookup):
        #self.pca = pca
        self.kmeans = kmeans
        self.lookup = lookup
        
    def predict(self,x):

        features = [f"feature_{i}" for i in range(130)]
        x = x.fillna(0)
        x = x[features].to_numpy().reshape((1, len(features)))
        x = self.pca.transform(x)
        x = self.kmeans.predict(x)[0]
        x = self.lookup[x]
        if x > 0: return 1
        else: return 0
        
```

Instantiate and run

```
submit_class = SubmitClassSL.load_from_disk("/kaggle/input/janestreet-model")

import janestreet
env = janestreet.make_env() # initialize the environment
iter_test = env.iter_test() # an iterator which loops over the test set

for (test_df, sample_prediction_df) in iter_test:
    sample_prediction_df.action = submit_class.predict(test_df)
    env.predict(sample_prediction_df)

```


