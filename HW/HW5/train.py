from args import get_parser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import os, random, math, argparse
import torchvision.transforms as transforms
import models
from dataset import get_loader
from torch.optim.lr_scheduler import *

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = None
train_list = []
val_list = []
# -------------------------------------------

def train(model, epoch, mode, data_loader, loss_func, optimizer, scheduler, generate_pred = False):
    
    total_loss = 0.0
    total_item = 0
    _pred_ = np.empty((1))
    _gt_ = np.empty((1))

    with tqdm(data_loader, unit="batch") as tepoch:

        for step, (data, label) in (enumerate(tepoch)):
            
            tepoch.set_description(f"Epoch {epoch},{mode}")
            
            if mode.lower() == "train":
                model.train()
                optimizer.zero_grad()
            
            #data = normalize(data.permute(2,0,1)).permute(1,2,0)

            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = loss_func(output, label)

            rmse_loss = torch.sqrt(loss)

            if mode.lower() == "train":
                loss.backward()
                optimizer.step()

            total_loss += rmse_loss.item() * output.shape[0]
            total_item += output.shape[0]
            tepoch.set_postfix({"RMSE_ls":rmse_loss.item()})

            if (generate_pred):
                output_np = output.cpu().detach().numpy()
                label_np = label.cpu().detach().numpy()
                _pred_ = np.append(_pred_, output_np.squeeze())
                _gt_ = np.append(_gt_, label_np.squeeze())

        #print("Final ", mode , " Loss: ", round(total_loss/total_item, 5))

        if generate_pred:
            return _pred_[1:], _gt_[1:]

        return total_loss / total_item



def main(args):
    
    global writer
    global device

    full_dataset, full_data_loader,train_loader, test_loader = get_loader(
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, 
        drop_last=False, args=args)

    if torch.cuda.is_available():
        device = torch.device("cuda:"+ str(args.device))

    model = models.myRNN(args)
    params = list(model.parameters())

    optimizer = torch.optim.Adam(params)


    learning_rate_scheduler = None
    
    mse_loss = torch.nn.MSELoss()
    model = model.to(device)

    print(model)
    print()
    print("---------- --------- ----------")
    print()

    min_val_ls = 100000000000000000000

    for epoch in range(args.num_epochs):

        model.train()
        train_loss = train(model = model, epoch = epoch, mode = "Train", data_loader = train_loader,
            loss_func = mse_loss, optimizer = optimizer, scheduler = learning_rate_scheduler)

        model.eval()
        with torch.no_grad():
            val_loss = train(model = model, epoch = epoch, mode = "Validation", data_loader = test_loader,
                loss_func = mse_loss, optimizer = optimizer, scheduler = None)

        train_list.append(train_loss)
        val_list.append(val_loss)

        print()

    print("***** Generating Predictions for full dataset *****")
    print()

    model.eval()
    with torch.no_grad():
        predictions, labels = train(generate_pred=True, model = model, epoch=0, mode="Pred_Gen", data_loader=full_data_loader, loss_func = mse_loss,
            optimizer = optimizer, scheduler = None)

    assert predictions.shape == labels.shape
    print()
    predictions = np.expand_dims(predictions, axis=1)
    labels = np.expand_dims(labels, axis=1)

    predictions_un_scld = full_dataset._inverse_scaler_(predictions).T.squeeze()
    labels_un_scld = full_dataset._inverse_scaler_(labels).T.squeeze()


    # prediction and ground truth plot

    plt.plot(predictions_un_scld, color='r', label="Model Predictions")
    plt.plot(labels_un_scld, color='b', label="Ground Truth")
    plt.legend(loc="upper left")
    plt.show()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = get_parser(parser)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    main(args)