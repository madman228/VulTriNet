import pickle, os, glob
import argparse
import pandas as pd
from collections import Counter
from collections import ChainMap

def sava_data(filename, data):
    print("开始保存数据至于：", filename)
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()

def load_data(filename):
    print("开始读取数据于：", filename)
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data

def parse_options():
    parser = argparse.ArgumentParser(description='Generate and split train datasettest_data.')
    parser.add_argument('-i', '--input', help='The path of a dir which consists of some pkl_files')
    parser.add_argument('-o', '--out', help='The path of output.', required=True)
    parser.add_argument('-n', '--num', help='Num of K-fold.', required=True)
    args = parser.parse_args()
    return args
    
def generate_dataframe(input_path, save_path):
    input_path = input_path + "/" if input_path[-1] != "/" else input_path
    save_path = save_path + "/" if save_path[-1] != "/" else save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    dic = []
    for type_name in os.listdir(input_path):
        dicname = input_path + type_name
        filename = glob.glob(dicname + "/*.pkl")
        for file in filename:
            data = load_data(file)
            dic.append({
                "filename": file.split("/")[-1].rstrip(".pkl"), 
                "length":   len(data[0]), 
                "data":     data, 
                "label":    0 if type_name == "No-Vul" else 1})
    final_dic = pd.DataFrame(dic)
    sava_data(save_path + "all_data.pkl", final_dic)

def gather_data(input_path, output_path):
    generate_dataframe(input_path, output_path)

def split_data(all_data_path, save_path, kfold_num):
    df_test = load_data(all_data_path)
    save_path = save_path + "/" if save_path[-1] != "/" else save_path
    seed = 0
    df_dict = {}
    train_dict = {i:{} for i in range(kfold_num)}
    test_dict = {i:{} for i in range(kfold_num)}
    from sklearn.model_selection import KFold
    kf = KFold(n_splits = kfold_num, shuffle = True, random_state = seed)
    for i in Counter(df_test.label.values):
        df_dict[i] = df_test[df_test.label == i]
        for epoch, result in enumerate(kf.split(df_dict[i])):
            train_dict[epoch][i]  = df_dict[i].iloc[result[0]]
            test_dict[epoch][i] =  df_dict[i].iloc[result[1]] 
    train_all = {i:pd.concat(train_dict[i], axis=0, ignore_index=True) for i in train_dict}
    test_all = {i:pd.concat(test_dict[i], axis=0, ignore_index=True) for i in test_dict}
    sava_data(save_path + "train.pkl", train_all)
    sava_data(save_path + "test.pkl", test_all)


from sklearn.model_selection import StratifiedKFold  
from imblearn.over_sampling import RandomOverSampler  

def split_data_imbalance(all_data_path, save_path, kfold_num):
    df_test = load_data(all_data_path)
    save_path = save_path + "/" if save_path[-1] != "/" else save_path
    seed = 0
    df_dict = {}
    train_dict = {i:{} for i in range(kfold_num)}
    test_dict = {i:{} for i in range(kfold_num)}

    skf = StratifiedKFold(n_splits=kfold_num, shuffle=True, random_state=seed)
    
    for label in df_test['label'].unique():
        label_data = df_test[df_test.label == label]
        splits = skf.split(label_data, label_data['label'])
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            train_dict[fold_idx][label] = label_data.iloc[train_idx]
            test_dict[fold_idx][label] = label_data.iloc[test_idx]

    ros = RandomOverSampler(random_state=seed)
    train_all, test_all = {}, {}
    
    for fold in range(kfold_num):
        train_fold = pd.concat(train_dict[fold].values(), ignore_index=True)
        test_fold = pd.concat(test_dict[fold].values(), ignore_index=True)
        
        X_train, y_train = train_fold.drop('label', axis=1), train_fold['label']
        X_res, y_res = ros.fit_resample(X_train, y_train)
        
        train_all[fold] = pd.DataFrame(X_res, columns=X_train.columns)
        train_all[fold]['label'] = y_res
        test_all[fold] = test_fold

    sava_data(save_path + "train.pkl", train_all)
    sava_data(save_path + "test.pkl", test_all)
    
def main():
    args = parse_options()
    input_path = args.input
    output_path = args.out
    kfold_num = args.num
    gather_data(input_path, output_path)
    kfold_num = int(kfold_num)
    split_data(output_path + "/all_data.pkl", output_path, kfold_num)
    #split_data_imbalance(output_path + "/all_data.pkl", output_path, kfold_num)
    

if __name__ == "__main__":
    main()