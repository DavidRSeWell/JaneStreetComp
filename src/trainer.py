"""
A trainer is a Class Object that glues all of the
pieces required for end to end training together
"""

import numpy as np
import pandas as pd


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


class BanditTrainer:
    def __init__(self,agent,train_env: TrainEnv, test_env: TrainEnv = None):
        self.agent = agent
        self.test_env = test_env
        self.train_env = train_env

    def eval(self,env):

        env.reset()

        tot_r = 0

        while True:

            state = env.state

            action = self.agent.predict(state)

            r, done = env.step(action)

            tot_r += r

            if done:
                break

        return tot_r

    def train(self,iters=10):

        train_rewards = []
        test_rewards = []

        for iter in range(iters):
            print(f"running iteration {iter}")
            print("Q Values")
            print(self.agent.Q)

            arm_selected_count = {x: 0 for x in range(self.agent.K)}
            self.agent.N = np.zeros(self.agent.K)

            self.train_env.reset()

            while True:

                state = self.train_env.state

                action = self.agent.act(self.train_env.curr_idx)

                r , done = self.train_env.step(action)

                arm = self.agent._labels[self.train_env.curr_idx]

                arm_selected_count[arm] += action

                self.agent.update(r,state)

                if done:
                    break

            print("Arm selected dist")
            print(arm_selected_count)
            print("N Values")
            print(self.agent.N)

            if self.test_env:
                train_rewards.append(self.eval(self.train_env))
                test_rewards.append(self.eval(self.test_env))

        return train_rewards,test_rewards

