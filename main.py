import argparse
from model import load_data, CNN_Classifier
import pickle
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time

def get_kfold_dataframe(pathname = "./data/", item_num = 0):
    pathname = pathname + "/" if pathname[-1] != "/" else pathname
    train_df = load_data(pathname + "train.pkl")[item_num]
    eval_df = load_data(pathname + "test.pkl")[item_num] 
    return train_df, eval_df

def load_pkl_files_in_directory(directory, label):
    data = []
    dic = []
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                dic.append({
                    "data":     data, 
                    "label":    label })
    final_dic = pd.DataFrame(dic)
    return final_dic
    


def main():
    item_num = 0
    hidden_size = 128
    data_path = '' 

    for item_num in range(5):
        train_df, eval_df = get_kfold_dataframe(pathname = data_path, item_num = item_num)
        print(len(train_df))
        print(len(eval_df))
        classifier = CNN_Classifier(result_save_path = data_path.replace("pkl", "results"), \
            item_num = item_num, epochs=100, hidden_size = hidden_size,max_len=225)
        classifier.preparation(
            X_train=train_df['data'],
            y_train=train_df['label'],
            X_valid=eval_df['data'],
            y_valid=eval_df['label'],
        )
        classifier.train()



if __name__ == "__main__":
    main()