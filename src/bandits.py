"""
Module containing all the
"""
import numpy as np

class SimpleKMeansBandit:

    def __init__(self,K,kmeans,tranform,threshold=0,eps=0.1):
        self.eps = eps
        self.threshold = threshold
        self.K = K
        self.N = np.zeros(K)
        self.Q = np.zeros(K)

        self._kmeans = kmeans
        self._labels = kmeans.labels_
        self._t = 0
        self._transform = tranform

    def act(self,idx):
        """
        Simple bandits dont care about no state
        :param state:
        :return:
        """

        self._t += 1
        if np.random.random() < self.eps:
            return np.random.randint(2) # random between pull or not pull
        else:
            #arm = self.get_arm_from_state(state)
            arm = self._labels[idx]
            # check if predicted is greater than some threshold
            ucbscore = self.ucb(arm,self._t)
            if ucbscore > self.threshold:
                return 1
            else:
                return 0

    def get_arm_from_state(self,state: np.array) -> int:
        """
        Look up the cluseter that the datapoint belongs to
        :param state:
        :return:
        """
        x = self._transform(state) # convert same was as was done in data.py
        cluster = self._kmeans.predict(x)[0]
        return cluster

    def predict(self,state):
        arm = self.get_arm_from_state(state)
        q = self.Q[arm]
        if q > 0: return 1
        else: return 0

    def ucb(self,action,t):
        if self.N[action] == 0:
            return self.threshold + 1
        return self.Q[action] + np.sqrt(np.log(t) / self.N[action])

    def update(self,r,state):
        a = self.get_arm_from_state(state)
        self.N[a] += 1
        update = (1.0 / self.N[a])*(r - self.Q[a])
        self.Q[a] += update