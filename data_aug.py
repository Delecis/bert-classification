# -*- coding: utf-8 -*-

import pandas as pd
import csv

from sklearn.model_selection import KFold
import os


def load_data(filename):
    # datas = pd.read_csv(filename).values.tolist()
    datas = pd.read_csv(filename, sep='\t', header=None, quoting=csv.QUOTE_NONE,error_bad_lines=False).values.tolist()
    return datas


def get_fold_data(datas, indexs):
    result = []
    for index in indexs:
        result.append(datas[index])
    return result


def add_id(datas):
    new_datas = []
    id = 0
    for data in datas:
        new_datas.append([id, data[0], data[1]])
        id += 1
    return new_datas


def add_id2(datas):
    new_datas = []
    id = 0
    for data in datas:
        new_datas.append([id, data[0]])
        id += 1
    return new_datas


def write_fold_data(datas, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(['id', 'body', 'label'])
        writer.writerows(datas)


def gen_kfold_data(datas, out_dir, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold = 0
    for train_index, dev_index in kf.split(datas):
        train_datas = get_fold_data(datas, train_index)
        dev_datas = get_fold_data(datas, dev_index)
        base_dir = os.path.join(out_dir, str(fold))
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        train_file = os.path.join(base_dir, 'train.csv')
        dev_file = os.path.join(base_dir, 'dev.csv')
        write_fold_data(train_datas, train_file)
        write_fold_data(dev_datas, dev_file)
        fold += 1


if __name__ == '__main__':
    # datas = load_data('../data/single_train.csv')
    # datas = add_id(datas)
    # gen_kfold_data(datas, '../data_AAAI/Kfold', k=5)

    datas = load_data('../DATA/train_shaming.csv')
    gen_kfold_data(datas, '../DATA/MAMI/Kfold_shaming', k=5)

    datas = load_data('../DATA/train_stereotype.csv')
    gen_kfold_data(datas, '../DATA/MAMI/Kfold_stereotype', k=5)

    datas = load_data('../DATA/train_objectification.csv')
    gen_kfold_data(datas, '../DATA/MAMI/Kfold_objectification', k=5)
    datas = load_data('../DATA/train_violence.csv')
    gen_kfold_data(datas, '../DATA/MAMI/Kfold_violence', k=5)




    # datas = load_data('../data_memotion/redata_B_humor.csv')
    # datas = add_id(datas)
    # gen_kfold_data(datas, '../data_memotion/reB_humor/Kfold', k=5)
    #
    # datas = load_data('../data_memotion/redata_B_sarcastic.csv')
    # datas = add_id(datas)
    # gen_kfold_data(datas, '../data_memotion/reB_sarcastic/Kfold', k=5)
    #
    #
    # datas = load_data('../data_memotion/redata_B_offensive2.csv')
    # datas = add_id(datas)
    # gen_kfold_data(datas, '../data_memotion/reB_offensive2/Kfold', k=5)
    #
    #
    # datas = load_data('../data_memotion/redata_B_motivational2.csv')
    # datas = add_id(datas)
    # gen_kfold_data(datas, '../data_memotion/reB_motivational2/Kfold', k=5)
    #
    # datas = load_data('../data_memotion/redata_C_humor.csv')
    # datas = add_id(datas)
    # gen_kfold_data(datas, '../data_memotion/reC_humor/Kfold', k=5)
    #
    # datas = load_data('../data_memotion/redata_C_sarcastic.csv')
    # datas = add_id(datas)
    # gen_kfold_data(datas, '../data_memotion/reC_sarcastic/Kfold', k=5)
    #
    #
    # datas = load_data('../data_memotion/redata_C_offensive.csv')
    # datas = add_id(datas)
    # gen_kfold_data(datas, '../data_memotion/reC_offensive/Kfold', k=5)
    #
    #
    # datas = load_data('../data_memotion/redata_C_motivational.csv')
    # datas = add_id(datas)
    # gen_kfold_data(datas, '../data_memotion/reC_motivational/Kfold', k=5)





    # df_test = load_data('../DATA/test_new.csv')
    # # df_test = add_id2(df_test)
    # write_fold_data(df_test, '../DATA/MAMI/test_new.csv')