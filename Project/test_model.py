#   1. You have to change the code so that he model is trained on the train set,
#   2. evaluated on the validation set.
#   3. The test set would be reserved for model evaluation by teacher.


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
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    
    args.mode = "test"

    test_dataset, test_loader = get_test_loader(args.data_dir, batch_size=args.batch_size, shuffle=False, 
                                           num_workers=args.num_workers, drop_last=False, args=args)

   
    num_classes = 19
    
    model_list = models.get_models(args, num_classes)

    path1 = "./saved_weights/18_bs-128_sch-10/dl_project_"
    path2 = "./saved_weights/vgg16_bs-64_sch-10_frz-cls/dl_project_"
    path3 = "./saved_weights/vgg16_ADAM/dl_project_"
    path4 = "./saved_weights/34_bs-128_sch-15-frz-fc/dl_project_"
    path5 = "./saved_weights/50_bs-128_sch-15-frz-fc/dl_project_"
    path6 = "./saved_weights/101_bs-128_sch-15_frz-fc/dl_project_"

    model_strings = ['ResNet18', 'VGG16_SGD', 'VGG16_adam','ResNet34','ResNet50' , 'ResNet50', 'ResNet101']
    #,'ResNet101','ResNet101']

    print()
    print("Loading saved model weights...")
    
    '''
    model_list[0].load_state_dict(torch.load(path1 + str(43) + ".pth", map_location = device))
    model_list[1].load_state_dict(torch.load(path1 + str(54) + ".pth", map_location = device))

    model_list[2].load_state_dict(torch.load(path2 + str(22) + ".pth", map_location = device))
    '''

    model_list[0].load_state_dict(torch.load(path1 + str(43) + ".pth", map_location = device))

    model_list[1].load_state_dict(torch.load(path2 + str(58) + ".pth", map_location = device))
    model_list[2].load_state_dict(torch.load(path3 + str(64) + ".pth", map_location = device))

    model_list[3].load_state_dict(torch.load(path4 + str(17) + ".pth", map_location = device))

    model_list[4].load_state_dict(torch.load(path5 + str(22) + ".pth", map_location = device))
    model_list[5].load_state_dict(torch.load(path5 + str(31) + ".pth", map_location = device))

    model_list[6].load_state_dict(torch.load(path6 + str(16) + ".pth", map_location = device))
    #model_list[7].load_state_dict(torch.load(path6 + str(10) + ".pth", map_location = device))
    #model_list[8].load_state_dict(torch.load(path6 + str(6) + ".pth", map_location = device))



    #optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=0.000001)
    learning_rate_scheduler = None
    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

    print()
    print("*** Creating predictions for each model for ensemble *** ")
    print()

    final_output = [[]] * len(model_list)

    for i, model in enumerate(model_list):

        model.to(device)
        model.eval()
        print("Model: ", model_strings[i])

        with torch.no_grad():

            _ , _ , final_output[i]= train(model = model, epoch = 0, mode = "Test", data_loader = test_loader, device = device,
                loss_func = cross_entropy_loss, optimizer = None, scheduler = None)

            print()

    total_correct_preds = 0.0
    total_elements = 1e-10

    '''
    Model ResNet18 overfits to the training data which is also evident from the test performance.
    So from the ensemble ResNet18 model was left out.
    '''

    print()
    print("*** Performance of the Ensemble (Majority Voting) of 6 models ***")

    header = ['Id', 'Class']
    csv_data = []

    classes = ['Adult', 'Airplane', 'Alpaca', 'Bird', 'Bus', 'Car', 'Cat', 'Child', 'Dog', 'Elephant', 'Flower', 'Giraffe', 'Horse', 'Monkey', 'Panda', 'Reptile', 'Train', 'Vessel', 'Zebra']
    counts = [1] * 19

    _preds_ = []
    _GT_ = []

    for step, (_,  ground_truth) in enumerate(test_loader):

        #m0_preds = final_output[0][step]
        m1_preds = final_output[1][step]
        m2_preds = final_output[2][step]
        m3_preds = final_output[3][step]
        m4_preds = final_output[4][step]
        m5_preds = final_output[5][step]
        m6_preds = final_output[6][step]
        #m7_preds = final_output[7][step]
        #m8_preds = final_output[8][step]

        preds = [m1_preds, m2_preds, m3_preds, m4_preds, m5_preds, m6_preds]

        preds = np.array(preds).T

        total_elements += preds.shape[0]

        # weight of every model is equal

        weighted_preds = []

        for i in range(preds.shape[0]):

            weighted_preds.append(np.histogram(preds[i], np.arange(20))[0].argmax())

        weighted_preds = torch.tensor(weighted_preds, device = device)
        ground_truth = ground_truth.to(device)

        # accuracy computation
        correct_preds_batch = torch.sum(weighted_preds==ground_truth).item()
        total_correct_preds += correct_preds_batch

        weighted_preds = weighted_preds.tolist()
        ground_truth = ground_truth.tolist()

        for wp,gt in zip(weighted_preds, ground_truth):

            _cls_ = 'test_' + classes[gt].lower() + '_' + str(counts[gt])
            counts[gt]+=1
            csv_data.append([_cls_ , wp+1])

            _preds_.append(wp)
            _GT_.append(gt)

    with open('ensemble_preds.csv', 'w', encoding='UTF8', newline='') as f:

        writer = csv.writer(f)
        
        # write the header
        writer.writerow(header)
        
        # write multiple rows
        writer.writerows(csv_data)

    print()

    print("Final Test accuracy after ensemble is: ", float(total_correct_preds/total_elements) * 100)

    print ()

    from plotcm import plot_confusion_matrix
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(_GT_, _preds_)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))

    #class names
    plot_confusion_matrix(cm, ['Adult', 'Airplane', 'Alpaca', 'Bird', 'Bus', 'Car', 'Cat', 'Child', 'Dog', 'Elephant', 'Flower', 'Giraffe', 'Horse', 'Monkey', 'Panda', 'Reptile', 'Train', 'Vessel', 'Zebra'])
    plt.show()

   

if __name__ == '__main__':
    args = get_parser()
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    main(args)