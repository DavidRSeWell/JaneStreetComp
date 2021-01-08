"""
A trainer is a Class Object that glues all of the
pieces required for end to end training together
"""

import numpy as np
import pandas as pd

from abc import ABC,abstractmethod
from sklearn.model_selection import train_test_split

from .utils import get_sub_config


class TrainEnv:
    def __init__(self,train_df: pd.DataFrame):
        self.curr_idx = 0
        self.train_df = train_df

    def reset(self):
        self.curr_idx = 0

    def reward(self,idx,action):

        row = self.train_df.iloc[idx]
        w = float(row["weight"])
        resp = float(row["resp"])

        return w*resp*action

    def step(self,action):

        r = self.reward(self.curr_idx,action)
        self.curr_idx += 1
        done = False
        if self.curr_idx == len(self.train_df) - 1:
            done = True
        return r , done

    @property
    def state(self):
        return self.train_df.iloc[self.curr_idx]


class Trainer(ABC):
    req_args = []
    op_args = []

    @classmethod
    @abstractmethod
    def load_from_config(cls,config):
        pass

    @abstractmethod
    def train(self):
        pass

    def eval(self,agent,env):

        env.reset()

        tot_r = 0

        while True:

            state = env.state

            action = agent.act(state)

            r, done = env.step(action)

            tot_r += r

            if done:
                break

        return tot_r


class BanditTrainer(Trainer):
    req_args = ["agent","data"]
    op_args = ["submission","iters"]

    @classmethod
    def load_from_config(cls,config):

        req_config = get_sub_config(config, cls.req_args)
        assert len(req_config) == len(cls.req_args)
        op_config = get_sub_config(config,cls.op_args)
        return cls(*req_config,*op_config)

    def __init__(self,agent,data,submission=False,iters=10):
        self.agent = agent

        self._data = data
        self._iters = iters
        self._submission = submission
        self._test_env = None
        self._train_env = None
        self._load_envs()

    def train(self):

        train_rewards = []
        test_rewards = []

        for iter in range(self._iters):
            print(f"running iteration {iter}")
            print("Q Values")
            print(self.agent.Q)

            arm_selected_count = {x: 0 for x in range(self.agent.K)}
            self.agent.N = np.zeros(self.agent.K)

            self._train_env.reset()

            while True:

                state = self._train_env.state

                action = self.agent.act(self._train_env.curr_idx)

                r , done = self._train_env.step(action)

                arm = self.agent._labels[self._train_env.curr_idx]

                arm_selected_count[arm] += action

                self.agent.update(r,state)

                if done:
                    break

            print("Arm selected dist")
            print(arm_selected_count)
            print("N Values")
            print(self.agent.N)

            if self._test_env:
                train_rewards.append(self.eval(self.agent,self._train_env))
                test_rewards.append(self.eval(self.agent,self._test_env))

        return train_rewards,test_rewards

    def _load_envs(self) -> None:

        self._train_env = TrainEnv(self._data)
        if not self._submission:
            # need to make a test env
            self._test_env = TrainEnv(self._data)


class SLTrainer(Trainer):

    req_args = ["agent","X","y"]
    op_args = ["submission", "iters","test_size"]

    @classmethod
    def load_from_config(cls, config) -> object:
        req_config = get_sub_config(config, cls.req_args)
        assert len(req_config) == len(cls.req_args)
        op_config = get_sub_config(config, cls.op_args)
        return cls(**req_config, **op_config)

    def __init__(self,agent,X,y,random_state=0,submission=False,iters=10,test_size=0.2):

        y = self.trans_y(y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        X, y, test_size = test_size, random_state = random_state)
        self._agent = agent
        self._iters = iters
        self._submission = submission

    def train(self) -> None:
        self._agent = self._agent.fit(self.X_train,self.y_train)

    def trans_y(self,y):
        y = np.where(y > 0, 1, y)
        y = np.where(y <= 0, 0, y)
        return y

    def test_predict(self) -> tuple:
        test_score = self._agent.score(self.X_test,self.y_test)
        train_score = self._agent.score(self.X_train,self.y_train)
        return train_score, test_score


