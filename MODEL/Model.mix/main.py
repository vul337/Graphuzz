import argparse
import json
import logging
import os
from time import time

import dgl
import tqdm
import torch
import dataloader
import torch.nn
import torch.nn.functional as F
import torch.utils.data
import torch.nn
from dgl.data import LegacyTUDataset
from dgl.dataloading import GraphDataLoader
from network import SAGNetworkHierarchical
from torch.utils.data import random_split
from utils import get_stats
import wandb
import random
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from torch.cuda.amp import autocast as autocast, GradScaler
from torch import autograd
from sklearn import preprocessing
import numpy as np

def scaler_norm(X):
    X = np.array(X).reshape(-1,1)
    robuster_scaler = preprocessing.RobustScaler(quantile_range=(25, 75))
    X_robuster = robuster_scaler.fit_transform(X)
    res = X_robuster
    #minmax_scaler=preprocessing.MinMaxScaler()
    #res=minmax_scaler.fit_transform(X_robuster)
    return res

def parse_args():
    parser = argparse.ArgumentParser(description="Self-Attention Graph Pooling")

    parser.add_argument(
        "--batch_size", type=int, default=64, help="batch size"
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="weight decay"
    )
    parser.add_argument(
        "--pool_ratio", type=float, default=0.3, help="pooling ratio"
    )
    parser.add_argument(
        "--conv_layers", type=int, default=5, help="number of conv layers"
    )
    parser.add_argument(
        "--line_layers", type=int, default=7, help="number of conv layers"
    )
    parser.add_argument("--hid_dim", type=int, default=256, help="hidden size")
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout ratio"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="max number of training epochs",
    )
    parser.add_argument(
        "--patience", type=int, default=100, help="patience for early stopping"
    )
    parser.add_argument(
        "--device", type=int, default=0, help="device id, -1 for cpu"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="hierarchical",
        choices=["hierarchical", "global", "double"],
        help="model architecture",
    )
    parser.add_argument(
        "--trainset", type=str, default="./trainset", help="path to trainset"
    )
    parser.add_argument(
        "--do_test", type=str, default="no", help="do test, without train"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./model.pth", help="path to save model"
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=1,
        help="print trainlog every k epochs, -1 for silent training",
    )
    parser.add_argument(
        "--num_trials", type=int, default=1, help="number of trials"
    )
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--model_path", type=str, default="./model.pth")
    parser.add_argument("--testset", nargs="+", type=str, default=[], help="path to testset")

    args = parser.parse_args()

    # device
    args.device = "cpu" if args.device == -1 else "cuda:{}".format(args.device)
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available, use CPU for training.")
        args.device = "cpu"

    # print every
    if args.print_every == -1:
        args.print_every = args.epochs + 1

    return args

def mape_score(y_true, y_pred):
    abs_perc_err = torch.abs((y_true - y_pred) / y_true)
    abs_perc_err[torch.isinf(abs_perc_err)] = 0
    mape = torch.mean(abs_perc_err) * 100
    return mape

#torch.autograd.set_detect_anomaly(True)

def train(model: torch.nn.Module, optimizer, criterion, trainloader, device):
    model.train()
    total_loss = 0
    total_loss_now = 0
    num_batch = 0
    scaler = GradScaler()
    for batch in tqdm.tqdm(trainloader):
        num_batch += 1
        optimizer.zero_grad()
        batch_graphs, batch_labels = batch
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.to(device)
        batch_graphs.ndata['attr'] = batch_graphs.ndata['attr'].type(torch.float16).to(device)
        batch_labels = batch_labels.type(torch.float16).to(device)
        #out = model(batch_graphs)
        #softmax_out = torch.softmax(out.type(torch.float32), dim=-1).to(device)
        #softmax_lab = torch.softmax(batch_labels.type(torch.float32), dim=-1).to(device)
        #print(out)
        #print(batch_labels)
        #loss = criterion(out, batch_labels)
        with autocast(dtype=torch.float16):
            out = model(batch_graphs)
            loss = criterion(out, batch_labels)
        scaler.scale(loss).backward()  # scaler实现的反向误差传播
        scaler.step(optimizer)  # 优化器中的值也需要放缩
        scaler.update()  # 更新scaler

        #with autograd.detect_anomaly():
        #    loss.backward()
        #    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2)
        #    optimizer.step()
        #loss.backward()
        #optimizer.step()
        #optimizer.zero_grad()

        total_loss += loss.item()
        total_loss_now += loss.item()
        #input("one batch done")
        if num_batch % 100 == 0:
            total_loss_now /= 100
            wandb.log({"train batch loss": total_loss_now})
            total_loss_now = 0
            print(out)
            print(batch_labels)
            print(loss)
        if loss.item() == float('inf') or loss.item() != loss.item():
            print(out)
            print(batch_labels)
            print(total_loss, loss.item(), num_batch)
            input("paused!")
    return total_loss / num_batch

def NORM_SOFTMAX(X):
    MAX = torch.max(X)
    MIN = torch.min(X)
    NORM = (X - MIN) / (MAX+1 - MIN)
    return torch.softmax(NORM, dim = -1)

def NORM(X):
    MAX = torch.max(X)
    MIN = torch.min(X)
    NORM = (X - MIN) / (MAX+1 - MIN)
    return NORM

def AVGNORM(X):
    MEAN = torch.mean(X)
    return NORM(X-MEAN)

def AVGDEC(X):
    MEAN = torch.mean(X)
    RES = (X - MEAN)
    return RES

@torch.no_grad()
def test(model: torch.nn.Module, criterion, loader, device):
    model.eval()
    loss = 0
    num_batch = 0

    predict = None
    labels = None

    for batch in tqdm.tqdm(loader):
        batch_graphs, batch_labels = batch
        num_batch += 1
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.to(device)
        with autocast():
            out = model(batch_graphs)
            loss += criterion(out, batch_labels).item()
        if predict == None:
            predict = out
            labels = batch_labels
        else:
            predict = torch.cat([predict, out], dim = -1)
            labels = torch.cat([labels, batch_labels], dim = -1)
        if loss == float('inf') or loss != loss:
            print(out)
            print(batch_labels)
    try:
        #loss = criterion(predict, labels)
        _r2 = r2_score(scaler_norm(labels.cpu().to(torch.float32)), scaler_norm(predict.cpu().to(torch.float32)))
        loss_softmax = criterion(predict.cpu().to(torch.float32), labels.cpu().to(torch.float32))
    except:
        __import__("traceback").print_exc()
        print(predict, labels, loss)
        _r2 = -9999999
        loss_softmax = -999999

    return loss / num_batch, _r2, loss_softmax

def get_test_res(model: torch.nn.Module, loader, device):
    model.eval()
    num_graphs = 0
    prev = []
    label = []
    for batch in tqdm.tqdm(loader):
        batch_graphs, batch_labels = batch
        num_graphs += batch_labels.size(0)
        batch_graphs = batch_graphs.to(device)
        batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
        out = model(batch_graphs)
        prev += out.tolist()
        label += batch_labels.tolist()
    return prev, label

def main(args):
    device = torch.device(args.device)
    print(device)
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    print("[+] Load dataset")
    
    trainset = dataloader.CDGDataset("DCG", args.trainset)
    testsets = []
    print(args.testset)
    for testset in args.testset:
        testsets.append(dataloader.CDGDataset("{}".format(testset).split("/")[-1].strip(), testset))
    
    # support batch graph.:

    train_loader = GraphDataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=False, persistent_workers=False, drop_last=True, prefetch_factor=7
    )
    
    test_loaders = []
    for testset in testsets:
        test_loaders.append(
            GraphDataLoader(
                testset, batch_size=args.batch_size, num_workers=8, pin_memory=False, persistent_workers=False, prefetch_factor=7
            )
        )

    # Step 2: Create model 
    print("[+] Create model")
    # =================================================================== #
    statics = trainset.statistics()
    num_feature, num_classes = statics[0:2]
    if os.path.exists(args.model_path):
        model = torch.load(args.model_path).to(device)
    else:
        model = SAGNetworkHierarchical(
            in_dim=num_feature,
            hid_dim=args.hid_dim,
            out_dim=num_classes,
            num_convs=args.conv_layers,
            pool_ratio=args.pool_ratio,
            dropout=args.dropout,
            num_lins=args.line_layers
        ).to(device)
    model = model.to(torch.float32)

    torch.compile(model)

    args.num_feature = int(num_feature)
    args.num_classes = int(num_classes)

    # Step 3: Create training components ===================================================== #
    print("[+] Create training components")
    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.SmoothL1Loss()
    #criterion = torch.nn.L1Loss()
    #criterion = mape_score
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)#, weight_decay=5e-11)
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-2) 

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.35, verbose=1, min_lr=1e-8,patience=30)
    
    wandb.watch(model, criterion)
    # Step 4: training epoches =============================================================== #
    print("[+] Start training")
    bad_cound = 0
    best_train_loss = float("inf")
    final_train_loss = 0.0
    best_epoch = 0
    train_times = []
    for e in range(args.epochs):
        s_time = time()
        train_loss = train(model, optimizer, criterion, train_loader, device)
        print("Epoch {}: train_loss={:.8f}".format(e + 1,  train_loss))
        train_times.append(time() - s_time)
        wandb.log({"train loss": train_loss})
        for test_loader in test_loaders:
            test_loss, _r2, loss_softmax = test(model, criterion, test_loader, device)
            print("Epoch {}: test_loss={:.8f}".format(e + 1,  test_loss))
            print("Epoch {}: test r2 score={:.8f}, mse_softmax={}".format(e + 1,  _r2, loss_softmax))
            wandb.log({
                "test loss {}".format(test_loader.dataset._name): test_loss, 
                "r2 score {}".format(test_loader.dataset._name): _r2, 
                "loss softmax {}".format(test_loader.dataset._name): loss_softmax}
            )
        scheduler.step(train_loss)
        if best_train_loss > train_loss:
            print("Epoch {}: final_train_loss={:.8f}, model saved!".format(e + 1,  train_loss))
            best_train_loss = train_loss
            final_train_loss = train_loss
            bad_cound = 0
            best_epoch = e + 1
            torch.save(model, args.save_dir)
        else:
            bad_cound += 1
            
    print(
        "Best Epoch {}, final train loss {:.8f}".format(
            best_epoch, final_train_loss
        )
    )
    return final_train_loss, sum(train_times) / len(train_times)

def do_test(random_attr_idx=None):
    device = torch.device(args.device)
    print(device)
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    print("[+] Load dataset")

    #dataset = LegacyTUDataset(args.dataset, raw_dir=args.dataset_path)
    #for i in range(len(dataset)):
    #    dataset.graph_lists[i] = dgl.add_self_loop(dataset.graph_lists[i])
    # add self loop. We add self loop for each graph here since the function "add_self_loop" does not
    
    testsets = []
    for testset in args.testset:
        testsets.append(dataloader.CDGDataset("{}".format(testset).split("/")[-1].strip(), testset, random_attr_idx=random_attr_idx))
    # support batch graph.:

    test_loaders = []
    for testset in testsets:
        test_loaders.append(
            GraphDataLoader(
                testset, batch_size=args.batch_size, num_workers=5, pin_memory=False, persistent_workers=False, prefetch_factor=4
            )
        )
        print(test_loaders[-1].dataset._name)

    # Step 2: Create model 
    print("[+] Create model")
    # =================================================================== #
    model = torch.load(args.model_path).to(device)
    #model = model.half()

    # Step 2: Do prediction
    print("[+] Do prediction")
    prev, label = get_test_res(model, test_loaders[0], device)
    idx = [i for i in range(0, len(label))]
    idx.sort(key=lambda a:label[a])
    
    prev_sorted = torch.tensor([prev[i] for i in idx], dtype=torch.float64)
    label_sorted = torch.tensor([label[i] for i in idx], dtype=torch.float64)

    prev_sorted = prev_sorted.tolist()
    label_sorted = label_sorted.tolist()
    #prev_sorted = prev_sorted.tolist()
    #label_sorted = label_sorted.tolist()

    #print([prev[i] for i in idx])
    #print([label[i] for i in idx])
    #print(prev_sorted)
    #print(label_sorted)

    return prev_sorted, label_sorted


if __name__ == "__main__":
    args = parse_args()
    if args.do_test != "no":
        prevs = []
        #for i in range(0, 19):
        #    prev_sorted, label_sorted = do_test(random_attr_idx=i)
        #    prevs.append(prev_sorted)
        for i in range(0, 1):
            prev_sorted, label_sorted = do_test(random_attr_idx=None)
            prevs.append(prev_sorted)
        with open("res.json", "w") as outf:
            json.dump([prevs, label_sorted], outf)
    else:
        res = []
        train_times = []
        for i in range(args.num_trials):
            wandb.init(
                # set the wandb project where this run will be logged
                project="DCFuzzProof",
                entity="sharkisai",
                # track hyperparameters and run metadata
                config={
                    "architecture": "GNN",
                    "dataset": "30 min with 481087 seed",
                    "epochs": 10000,
                    "learning_rate": args.lr,
                    "conv_layers": args.conv_layers,
                    "pool_ratio": args.pool_ratio,
                    "hid_dim": args.hid_dim,
                    "batch_size": args.batch_size,
                    "line_layers": args.line_layers,
                    "optimizer": "AdamW",
                    "scheduler": "ReduceLROnPlateau",
                    "scheduler factor": 0.35
                }
            )
            print("Trial {}/{}".format(i + 1, args.num_trials))
            final_test_loss, train_time = main(args)
            res.append(final_test_loss)
            train_times.append(train_time)

        mean, err_bd = get_stats(res)
        print("mean acc: {:.8f}, error bound: {:.8f}".format(mean, err_bd))
        
        wandb.finish()
