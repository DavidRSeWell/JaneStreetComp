import os
import pandas as pd
import time


#import janestreet
from src.data import JaneData

def run(data_dir):
    data = JaneData(data_dir)

if __name__ == "__main__":

    st = time.time()
    data_dir = "/Users/davidsewell/MLData/JaneStreet"
    run(data_dir)
    et = time.time()
    rt = (et - st) / 60.0
    print(f"Runtime {rt}")
