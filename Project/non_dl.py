from args import get_parser
import torch
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, vgg16, vgg19, inception_v3
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import os, random, math

import models

from dataset import get_loader, get_test_loader, get_original_dataset
from torch.optim.lr_scheduler import StepLR

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def svm_lin():

	classifier_svm_lin = SVC(kernel='linear', probability=False)

	classifier_svm_lin.fit(train_data, train_label)

	test_pred_SVM_lin = classifier_svm_lin.predict(test_data)

	correct_preds_lin = np.sum(np.equal(test_pred_SVM_lin, test_label))

	print("*** SVM linear kernel ***")
	print("Test Accuracy : ", correct_preds_lin/len(test_data))
	print()


def svm_rbf():

	classifier_svm_rbf = SVC(kernel='rbf', probability=False)

	classifier_svm_rbf.fit(train_data, train_label)

	test_pred_SVM_rbf = classifier_svm_rbf.predict(test_data)

	correct_preds_rbf = np.sum(np.equal(test_pred_SVM_rbf, test_label))


	print("*** SVM rbf kernel ***")
	print("Test Accuracy : ", correct_preds_rbf/len(test_data))
	print()


def RF(n_est):

	classifier_RF_ = RandomForestClassifier(n_estimators=n_est)
	classifier_RF_.fit(train_data,train_label)

	test_pred_RF = classifier_RF_.predict(test_data)

	correct_preds_RF = np.sum(np.equal(test_pred_RF, test_label))


	print("*** RF classifier with n_estimators = " + str(n_est) + " ***")
	print("Test Accuracy : ", correct_preds_RF/len(test_data))
	print()


global train_data, train_label, test_data, test_label

args = get_parser()

num_classes = 19

dataset , train_loader = get_original_dataset(args.data_dir, batch_size=args.batch_size, shuffle=True, 
	num_workers=args.num_workers, drop_last=False, args=args, algorithm="SVM")

args.mode = "test"

test_dataset, test_loader = get_test_loader(args.data_dir, batch_size=args.batch_size, shuffle=False, 
                                           num_workers=args.num_workers, drop_last=False, args=args, algorithm="SVM")
args.mode = "train"

train_data , train_label = next(iter(train_loader))

train_data = train_data.tolist()
train_label = train_label.tolist()

test_data, test_label = next(iter(test_loader))

test_data = test_data.tolist()
test_label = test_label.tolist()

print("Train data size : ",len(train_data))
print("Test data size : ",len(test_data))

exi()

print()

svm_lin()
svm_rbf()

RF(n_est=50)
RF(n_est=100)
RF(n_est=200)
RF(n_est=300)
RF(n_est=500)


print()



'''

*** RF classifier with n_estimators=50 ***
Test Accuracy :  0.17543859649122806

*** RF classifier with n_estimators=100 ***
Test Accuracy :  0.21754385964912282

*** RF classifier with n_estimators=200 ***
Test Accuracy :  0.20350877192982456

*** RF classifier with n_estimators=300 ***
Test Accuracy :  0.22280701754385965

*** RF classifier with n_estimators=500 ***
Test Accuracy :  0.22807017543859648


'''


