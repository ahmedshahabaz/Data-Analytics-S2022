import torch
from torchvision import transforms
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import os, glob
import pickle, json
import numpy as np
import random, collections
from PIL import Image
from sklearn.model_selection import train_test_split
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------

def get_loader(data_dir, batch_size, shuffle, num_workers, drop_last=False, args=None, algorithm="DL"):

    '''
        data loader function
    '''

    dataset = ImageDataset(data_dir=data_dir, args=args, algorithm=algorithm)

    train_indices = torch.multinomial(torch.tensor(dataset.sample_weights), num_samples=len(dataset)+600, 
        replacement=True, generator=torch.Generator()).tolist()

    val_indices = []

    all_indices = list(range(2223))

    for i_ in all_indices:
        
        if(i_ not in train_indices):
            val_indices.append(i_)

    random.shuffle(val_indices)
    random.shuffle(train_indices)

    '''
    sample_weights = [0] * len(dataset)

    for iid, (_, label) in enumerate(dataset):

        sample_weights[iid] = class_weights[label]
    
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),replacement = True)
    '''

    #train_indices , val_indices = train_test_split(indices, test_size = 150 , shuffle=True)


    #val_set_size = 150
    #train_set_size = len(dataset) - 150 #int(len(dataset) * 0.9)
    #train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_set_size, val_set_size])

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices, generator=None)

    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices, generator=None)

    train_batch_size = batch_size
    val_batch_size = batch_size

    if algorithm=="svm" or algorithm == "SVM":
        
        train_batch_size = len(train_indices)
        val_batch_size = len(val_indices)

    train_data_loader = torch.utils.data.DataLoader(dataset=dataset, sampler = train_sampler,
                                              batch_size=train_batch_size, num_workers=num_workers,
                                              drop_last=drop_last,  pin_memory=True)

    val_data_loader = torch.utils.data.DataLoader(dataset=dataset, sampler = val_sampler,
                                              batch_size=val_batch_size, num_workers=num_workers,
                                              drop_last=drop_last,  pin_memory=True)
    print()
    print ("Data loader in {} mode!".format(args.mode))
    print ("Number of classes: {}.".format(dataset.num_classes))
    print ("Train Data size: {}.".format(len(train_indices)))
    print ("Validation Data size: {}.".format(len(val_indices)))
    print (" ---------- --------- ---------- \n")

    return dataset, train_data_loader, val_data_loader

# -------------------------------------------

def get_original_dataset(data_dir, batch_size, shuffle, num_workers, drop_last=False, args=None, algorithm="DL"):

    '''
        data loader function that returns original imbalanced dataset
    '''

    dataset = ImageDataset(data_dir=data_dir, args=args, algorithm=algorithm)

    if algorithm=="svm" or algorithm == "SVM":

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset),
            num_workers = num_workers, shuffle = True)

        print()
        print ("Data loader in {} mode!".format(args.mode))
        print ("Number of classes: {}.".format(dataset.num_classes))
        print ("Train Data size: {}.".format(len(dataset)))

        return dataset, data_loader


    dataset_size = len(dataset)

    test_set_size = int(dataset_size * .1)
    train_set_size = dataset_size - test_set_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_set_size, test_set_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
        num_workers = num_workers, shuffle = True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
        num_workers = num_workers, shuffle = True)

    print()
    print ("Data loader in {} mode!".format(args.mode))
    print ("Number of classes: {}.".format(dataset.num_classes))
    print ("Train Data size: {}.".format(len(train_dataset)))
    print ("Validation Data size: {}.".format(len(test_dataset)))
    print (" ---------- --------- ---------- \n")

    return dataset, train_loader, test_loader


# -------------------------------------------

def get_test_loader(data_dir, batch_size, shuffle, num_workers, drop_last=False, args=None, algorithm="DL"):

    test_dataset = ImageDataset(data_dir=data_dir, args=args, algorithm=algorithm)

    if algorithm == 'SVM' or algorithm == 'svm':

        test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle = False,
            batch_size=len(test_dataset), num_workers=num_workers,
            drop_last=drop_last,  pin_memory=True)

    else:
        test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle = False,
            batch_size=batch_size, num_workers=num_workers,
            drop_last=drop_last,  pin_memory=True)


    print()
    print ("Data loader in {} mode!".format(args.mode))
    print ("Number of classes: {}.".format(test_dataset.num_classes))
    print ("Test Data size: {}.".format(len(test_dataset)))
    print (" ---------- --------- ---------- \n")

    return test_dataset, test_data_loader

# -------------------------------------------

"""
   Auxilary data functions
"""


def transformation_functions(args, mode="train", algorithm="DL"):
    '''
        apply transfomrations to input images.
        the type of transformation (augmentations) used is different at train and evaluation time.
        mode: can be used to explicitly specify separate operations for train and evaluation mode (e.g. augmentation)
    '''

    if algorithm == "svm" or algorithm == "SVM":
        transforms_list = [
        transforms.Resize((args.image_size)),
        transforms.CenterCrop(args.crop_size)
        ]
    
    else:
        if mode == "train" or mode == "Train":
            transforms_list = [
            transforms.Resize((args.image_size)),
            transforms.CenterCrop(args.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            #transforms.RandomAutocontrast(),
            #transforms.RandomEqualize(),
            transforms.RandomApply(torch.nn.ModuleList([
                transforms.ColorJitter(),
                transforms.GaussianBlur(15),
                ]), p = 0.3),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
            ]

        else:
            transforms_list = [
            transforms.Resize((args.image_size)),
            transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
            ]                                                                                                                                                                                                                                   

    return transforms.Compose(transforms_list) 

# -------------------------------------------

def create_class_indice(names):

    idx_to_class = {}
    class_to_idx = {}
    idx = 0
    for name in sorted(names):
        class_to_idx[name] = idx
        idx_to_class[idx] = name
        idx += 1

    return class_to_idx, idx_to_class


# -------------------------------------------

class ImageDataset(data.Dataset):
    """
        Dataloader to generate dataset
    """

    def __init__(self, data_dir="data", transforms=None, args=None, algorithm="DL"):
        '''
            function to initialize dataset.
        '''

        self.transform = transformation_functions(args, args.mode)

        self.algorithm = algorithm

        if args.mode == "train":
            dataset_split_path = os.path.join(data_dir, "train")
        elif args.mode == "test":
            dataset_split_path = os.path.join(data_dir, "test")

        
        dataset_class_folders = os.listdir(dataset_split_path)

        self.id_to_path = {}
        self.id_to_class = {}
        self.classes_to_count = {}
        id_to_sample_weight = {}

        iid = 0

        for class_folder in dataset_class_folders:

            img_files = [files for _,_,files in os.walk("./" + dataset_split_path + "/" + class_folder + "/")]
            sample_weight_ = 1/(len(img_files[0]))
            self.classes_to_count[class_folder] = len(img_files[0])

            for file in img_files[0]:

                path = "./" + dataset_split_path + "/" + class_folder + "/" + file         
                self.id_to_path[iid] = path
                self.id_to_class[iid] = class_folder
                id_to_sample_weight[iid] = sample_weight_
                iid += 1

        self.class_to_idx, self.idx_to_class = create_class_indice(self.classes_to_count)
        # get ids of all samples in this split of the dataset        
        self.ids = list(self.id_to_path.keys())
        self.num_classes = len(self.classes_to_count)
        self.dataset_size = sum(self.classes_to_count.values())

        self.num_inputs = 1 #args.batch_size
        self.num_targets = self.num_inputs
        self.sample_weights = list(id_to_sample_weight.values())


    def get_num_classes(self):
        return self.num_classes

    def __getitem__(self, iid):
        """
            Returns image, class index
        """
        iid = self.ids[iid]
        img_path = self.id_to_path[iid]
        class_name = self.id_to_class[iid]
        class_idx = self.class_to_idx[class_name]
        # load image
        image_input = Image.open(self.id_to_path[iid]).convert('RGB')

        if self.transform is not None:
            image_input = self.transform(image_input)
        else:
            image_input = transforms.Compose(transforms.ToTensor())(image_input)


        if self.algorithm=='svm' or self.algorithm=='SVM':

            image_input = image_input.flatten()

        
        return image_input, class_idx

    def __len__(self):
        return len(self.ids)

# -------------------------------------------