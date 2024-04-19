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

random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path= "ToNv2/"
PATH_TON_DATASET = "/data176/privatecloud/data/autodl-container-1b164bbe69-af1a6422-storage/reduced_ids_data/toniot.csv"#数据集路径

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
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    # trainset = torchvision.datasets.EMNIST(
    #     root=dir_path+"rawdata", split='digits', train=True, download=True, transform=transform)
    # testset = torchvision.datasets.EMNIST(
    #     root=dir_path+"rawdata", split='digits', train=False, download=True, transform=transform)

    df = pd.read_csv(PATH_TON_DATASET)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)    
    df.dropna(inplace=True)
    #target_column= ["Attack",'IPV4_SRC_ADDR', 'IPV4_DST_ADDR']#选择不要的特征，二分类需要抛弃的列
    target_column= ["Label",'IPV4_SRC_ADDR', 'IPV4_DST_ADDR']#选择不要的特征，多分类需要抛弃的列
    df = df.drop(columns=target_column)
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
 
    scaler = StandardScaler()
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

    generate_dataset(dir_path, num_clients, niid, balance, partition)