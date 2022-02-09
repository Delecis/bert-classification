
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
# import logging
# logging.basicConfig(level=logging.ERROR)
# cmd = "./main.sh"
# data = os.popen(cmd)
# print(data.read())
import pandas as pd
df_train = pd.read_csv('../data/train.csv')
print(len(df_train))
print(df_train[-1:])