from plotcm import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from args import get_parser
import torch
from torchvision.models import resnet18, resnet50, resnet101, resnet152, vgg16, vgg19, inception_v3
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import os, random, math
import models
from dataset import get_test_loader
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
from train import train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):

    args.mode = "test"

    test_dataset, test_loader = get_test_loader(args.data_dir, batch_size=args.batch_size, shuffle=False, 
                                           num_workers=args.num_workers, drop_last=False, args=args)
    num_classes = 19

    # all_target = ground truth
    # all_pred = prediction of the model

    model_50 = resnet101(pretrained = False)
    model_50.fc = nn.Linear(2048,num_classes)

    model_50_1 = resnet50(pretrained = False)
    model_50_1.fc = nn.Linear(2048,num_classes)

    path1 = "./saved_weights/50_bs-128_sch-15-frz-fc/dl_project_"
    path6 = "./saved_weights/101_bs-128_sch-15_frz-fc/dl_project_"
    model_50.load_state_dict(torch.load(path6 + str(16) + ".pth", map_location = device))

    path2 = "./saved_weights/50_bs-128_sch-15-frz-fc_full_ds/dl_project_"
    model_50_1.load_state_dict(torch.load(path2 + str(63) + ".pth", map_location = device))

    model_list = [model_50, model_50_1]

    final_output = [[]] * len(model_list)

    learning_rate_scheduler = None
    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

    for i, model in enumerate(model_list):

        model.to(device)
        model.eval()

        with torch.no_grad():

            _ , _ , final_output[i]= train(model = model, epoch = 0, mode = "Test", data_loader = test_loader, device = device,
                loss_func = cross_entropy_loss, optimizer = None, scheduler = None)

            print()

    m1_preds = []
    m2_preds = []
    GT = []

    for step, (_,  ground_truth) in enumerate(test_loader):

        for f1, f2 in zip(final_output[0][step], final_output[1][step]):

            m1_preds.append(f1)
            m2_preds.append(f2)

        torch.reshape(ground_truth, (ground_truth.shape))

        ground_truth = ground_truth.tolist()

        for gt in ground_truth:
            GT.append(gt)

    print()

    cm1 = confusion_matrix(GT, m1_preds)
    cm2 = confusion_matrix(GT, m2_preds)
    #print(cm1)
    #print()
    #print(cm2)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))

    #class names
    plot_confusion_matrix(cm1, ['Adult', 'Airplane', 'Alpaca', 'Bird', 'Bus', 'Car', 'Cat', 'Child', 'Dog', 'Elephant', 'Flower', 'Giraffe', 'Horse', 'Monkey', 'Panda', 'Reptile', 'Train', 'Vessel', 'Zebra'])
    plt.show()
    plt.figure(figsize=(10,10))

    plot_confusion_matrix(cm2, ['Adult', 'Airplane', 'Alpaca', 'Bird', 'Bus', 'Car', 'Cat', 'Child', 'Dog', 'Elephant', 'Flower', 'Giraffe', 'Horse', 'Monkey', 'Panda', 'Reptile', 'Train', 'Vessel', 'Zebra'])
    plt.show()





    print ()

if __name__ == '__main__':
    args = get_parser()
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    main(args)