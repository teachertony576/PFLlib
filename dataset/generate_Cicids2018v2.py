# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
import pandas as pd
import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path= "Cicids2018v2/"
PATH_TON_DATASET = "/data176/privatecloud/data/autodl-container-1b164bbe69-af1a6422-storage/ids_data/NF-CSE-CIC-IDS2018-v2.csv"#数据集路径

# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # Get v2 data


    df = pd.read_csv(PATH_TON_DATASET)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)    
    df.dropna(inplace=True)
    #target_column= ["Attack",'IPV4_SRC_ADDR', 'IPV4_DST_ADDR']#选择不要的特征，二分类需要抛弃的列
    target_column= ["Label",'IPV4_SRC_ADDR', 'IPV4_DST_ADDR']#选择不要的特征，多分类需要抛弃的列

    ddos=df[df['Attack'].str.contains("ddos",case=False)]
    dos=df[df['Attack'].str.contains("dos",case=False)]
    df.drop(index=dos.index,inplace=True)
    dos.drop(index=ddos.index,inplace=True)

    brute=df[df['Attack'].str.contains("brute",case=False)]
    df.drop(index=brute.index,inplace=True)
    print(df.Attack.value_counts())
    print("before subsampling")
    print(dos.Attack.value_counts())
    print(ddos.Attack.value_counts())
    print(brute.Attack.value_counts())

    # grouped = dos.groupby(dos.Attack)
    # dos_attacks=[ grouped.get_group(attack).sample(9512) for attack in dos.Attack.unique() ]
    dos=pd.concat(objs=[dos])

    # grouped = ddos.groupby(ddos.Attack)
    # ddos_attacks=[ grouped.get_group(attack).sample(13828) for attack in ddos.Attack.unique() ]
    ddos=pd.concat(objs=[ddos])

    # grouped = brute.groupby(brute.Attack)
    # brute_attacks=[ grouped.get_group(attack).sample(1212) for attack in brute.Attack.unique() ]
    brute=pd.concat(objs=[brute])


    print("after subsampling")
    print(dos.Attack.value_counts())
    print(ddos.Attack.value_counts())
    print(brute.Attack.value_counts())

    dos.Attack="DoS"
    ddos.Attack="DDoS"
    brute.Attack="Brute Force"
    df=pd.concat(objs=[df, dos,ddos,brute])



    df = df.drop(columns=target_column)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)    
    print(df.isna().any(axis=1).sum(), "rows with at least one NaN to remove")
    df.dropna(inplace=True)
    df = df.drop_duplicates()
    label_encoder = LabelEncoder()
    # 将Attack列的字符串标签转换为数字标签
    df['Attack'] = label_encoder.fit_transform(df['Attack'])
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1:].values
    data = {
        "X": X,
        "y": y,
    }
 
    scaler = MinMaxScaler()#StandardScaler会产生Nan值，选择MinMaxScaler
    scaler.fit(data["X"])

    data["X"] = scaler.transform(data["X"])

    dataset_image = []
    dataset_label = []

    dataset_image = np.array(data["X"])
    dataset_label = np.array(data["y"].flatten())#转为一维列表

    num_classes = len(set(tuple(dataset_label)))
    print(f'Number of classes: {num_classes}')

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    # niid = True
    # balance = True
    # partition = "pat"

    generate_dataset(dir_path, num_clients, niid, balance, partition)