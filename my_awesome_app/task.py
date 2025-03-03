"""my-awesome-app: A Flower / PyTorch app."""

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from .model import HaiDataset, StackedGRU, normalize, inference , train , put_labels , fill_blank
import pandas as pd
import pickle

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


N_HIDDENS = 100
N_LAYERS = 3
BATCH_SIZE = 512
THRESHOLD = 0.026   

class Net(torch.nn.Module):
    def __init__(self, n_tags):
        super().__init__()
        self.rnn = torch.nn.GRU(
            input_size=n_tags,
            hidden_size=N_HIDDENS,
            num_layers=N_LAYERS,
            bidirectional=True,
            dropout=0,
        )
        self.fc = torch.nn.Linear(N_HIDDENS * 2, n_tags)

    def forward(self, x):
        x = x.transpose(0, 1)  
        self.rnn.flatten_parameters()
        outs, _ = self.rnn(x)
        out = self.fc(outs[-1])
        return x[0] + out


fds = None  # Cache FederatedDataset


# def load_data(partition_id: int, num_partitions: int):
#     """Load partition CIFAR10 data."""
#     # Only initialize `FederatedDataset` once
#     global fds
#     if fds is None:
#         partitioner = IidPartitioner(num_partitions=num_partitions)
#         fds = FederatedDataset(
#             dataset="uoft-cs/cifar10",
#             partitioners={"train": partitioner},
#         )
#     partition = fds.load_partition(partition_id)
#     # Divide data on each node: 80% train, 20% test
#     partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
#     pytorch_transforms = Compose(
#         [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     )

#     def apply_transforms(batch):
#         """Apply transforms to the partition from FederatedDataset."""
#         batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
#         return batch

#     partition_train_test = partition_train_test.with_transform(apply_transforms)
#     trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
#     testloader = DataLoader(partition_train_test["test"], batch_size=32)
#     return trainloader, testloader

TIMESTAMP_FIELD = "timestamp"
IDSTAMP_FIELD = 'id'
ATTACK_FIELD = "Attack"

WINDOW_GIVEN = 89
WINDOW_SIZE = 90




def train_client(net, device,partition_id):
    """Train the model on the training set."""
    net.to(device) 
    train_path = f"/home/kurose/fdl/data/train/train{partition_id}.csv"
    
    TRAIN_DF_RAW = pd.read_csv(train_path).rename(columns=lambda x: x.strip())
    VALID_COLUMNS_IN_TRAIN_DATASET = TRAIN_DF_RAW.columns.drop([TIMESTAMP_FIELD])
    
    with open("/home/kurose/fdl/my-awesome-app/my_awesome_app/tag_min_max.pkl", "rb") as f:
       data = pickle.load(f)

    TAG_MIN = data["TAG_MIN"]
    TAG_MAX = data["TAG_MAX"]
    
    TRAIN_DF = normalize(TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET] , TAG_MAX, TAG_MIN).ewm(alpha=0.9).mean()

    HAI_DATASET_TRAIN = HaiDataset(TRAIN_DF_RAW[TIMESTAMP_FIELD], TRAIN_DF, stride=10)
    

    BEST_MODEL, LOSS_HISTORY = train(HAI_DATASET_TRAIN, net , BATCH_SIZE, 32, device)
    print("LOSS",BEST_MODEL["loss"],"EPOCH", BEST_MODEL["epoch"])


    return BEST_MODEL["loss"]


def loss_helper(net,device):
    net.to(device)
    # VALIDATION_DATASET = ["/home/kurose/fdl/data/test/test1.csv","/home/kurose/fdl/data/test/test2.csv","/home/kurose/fdl/data/test/test3.csv","/home/kurose/fdl/data/test/test0.csv"]
    VALIDATION_DATASET = ["/home/kurose/fdl/data/test/test0.csv" ,"/home/kurose/fdl/data/test/test1.csv" ]
    VALIDATION_DF_RAW = pd.concat(
    [pd.read_csv(file).rename(columns=lambda x: x.strip()) for file in VALIDATION_DATASET], 
    ignore_index=True
    )
    
    VALID_COLUMNS_IN_TRAIN_DATASET = VALIDATION_DF_RAW.columns.drop([TIMESTAMP_FIELD])
    with open("/home/kurose/fdl/my-awesome-app/my_awesome_app/tag_min_max.pkl", "rb") as f:
       data = pickle.load(f)

    TAG_MIN = data["TAG_MIN"]
    TAG_MAX = data["TAG_MAX"]
    
    VALIDATION_DF = normalize(VALIDATION_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET],TAG_MAX,TAG_MIN)

    HAI_DATASET_VALIDATION = HaiDataset(
    VALIDATION_DF_RAW[TIMESTAMP_FIELD], VALIDATION_DF, attacks=VALIDATION_DF_RAW[ATTACK_FIELD]
    )

    net.eval()
    CHECK_TS, CHECK_DIST, CHECK_ATT = inference(HAI_DATASET_VALIDATION, net, BATCH_SIZE,device)
    ANOMALY_SCORE = np.mean(CHECK_DIST, axis=1)

    LABELS = put_labels(ANOMALY_SCORE, THRESHOLD)
    ATTACK_LABELS = put_labels(np.array(VALIDATION_DF_RAW[ATTACK_FIELD]), threshold=0.5)
    FINAL_LABELS = fill_blank(CHECK_TS, LABELS, np.array(VALIDATION_DF_RAW[TIMESTAMP_FIELD]))
    TaP = precision_score(ATTACK_LABELS, FINAL_LABELS)
    TaR = recall_score(ATTACK_LABELS, FINAL_LABELS)
    f1 = f1_score(ATTACK_LABELS, FINAL_LABELS)

    print("RECALL :",TaR,"\nPrecision",TaP,"\nF1",f1 ) 
    
    return TaP, TaR, f1

def test(net, device):
    """Validate the model on the test set."""
    net.to(device)
    TaP, TaR, f1 = loss_helper(net,device)

    return TaP, f1


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
