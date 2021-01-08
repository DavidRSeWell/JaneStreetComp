from abc import ABC,abstractmethod

from sklearn.linear_model import LogisticRegression

from .utils import get_sub_config


class SLAgent(ABC):
    """
    Base Class for supervised learning agents.
    A SL agent is a traditional SL algorithm (regression, svm, ect..)
    but it can make decesions based on a prediction it makes
    """
    @abstractmethod
    def act(self,x):
        pass

    @abstractmethod
    def score(self,X,y):
        pass

    @abstractmethod
    def fit(self,X,y):
        pass


class LogisticRegAgent(SLAgent):

    op_args = ["model_config","threshold"]

    @classmethod
    def load_from_config(cls, config):
        op_config = get_sub_config(config, cls.op_args)
        return cls(**op_config)

    def __init__(self,model_config={},threshold=0):
        self._model = LogisticRegression(**model_config)
        self._threshold = threshold

    def act(self,x):
        y_hat = self._model.predict(x)
        if (y_hat > self._threshold):
            return 1
        else:
            return 0

    def score(self,X,y):
        return self._model.score(X,y)

    def fit(self,X,y):
        return self._model.fit(X,y)