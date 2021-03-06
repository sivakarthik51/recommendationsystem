import numpy as np
import pandas as pd
import os

PATH = "../data/goodreads/"
DATA_FILE = "ratings.csv"
cols_to_use = ['user_id', 'book_id']
#cols_to_use = ['user','movie']


df = pd.read_csv(os.path.join(PATH, DATA_FILE), usecols=cols_to_use)
print(f"len of entire df = {len(df)}")
train_df, test_df = pd.DataFrame(), pd.DataFrame()
train_ratio = 0.9
for k, g in df.groupby(cols_to_use[0]):
    num_items = len(g)
    items = g.sample(frac=1)
    train_items = int(train_ratio*num_items)
    train_df = pd.concat([train_df, items[0:train_items]])
    test_df = pd.concat([test_df,items[train_items:]])
    # print(num_items, len(train_df[k]), len(test_df[k]))
train_df.reset_index(inplace=True, drop=True)
test_df.reset_index(inplace=True, drop=True)

assert((df.sort_values(by=cols_to_use).reset_index(drop=True) == pd.concat([train_df, test_df]).sort_values(by=cols_to_use).reset_index(drop=True)).all().all())

train_df.to_csv(os.path.join(PATH, os.path.splitext(DATA_FILE)[0]+"_train.csv"))
test_df.to_csv(os.path.join(PATH, os.path.splitext(DATA_FILE)[0]+"_test.csv"))


