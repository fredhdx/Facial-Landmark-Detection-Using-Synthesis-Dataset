import os, sys
import numpy as np
from tqdm.auto import tqdm

import torch
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim
import datetime

from helper.dataGenerater import generate_train_data
from xception.xception import LandmarksDataset, Preprocessor, XceptionNet
import json
import pickle

import argparse

#########################################
########### required file system #######
#######################################
## Source Code 
## /train.py ###########################
##     /train_set1 ####################
##          train.json ###############
##          model.epoch #############
##          model.pt ###############
##          traindata.pickle ######
##      /trian_set2 #################
##      /trian_set3 #################
#######################################
# run: python3 train.py --file train_set1 --resume (optional)
#######################################
#  train.json: training and model parameters
#  model.pt: checkpoint file
#  model.epoch: tracking text file
#  traindata.pickle: pickle for generated traindata list
#######################################
#######################################
## Data storage
## /mydata
##      /ibug_300W_large_face_landmark_dataset <-- 300W real face dataset
##      /dataset_5000 <-- MS Synthesis dataset
##      labels_ibug_300W_train.xml
##      labels_ms_synthesis_dlib.xml
##      labels_ms_synthesis_landmark.xml
##      labels_ms_synthesis_mix.xml
#######################################

#######################################
######## Evaluation ###################
## Please use eval.ipynb notebook ####
######################################

######################################
######################################
## Example train.json
# {
#     "data_dir": "/absolute/path/project/dataset", <-- point to dataset folder
#     "bbox_option": "mix",  <-- choose how bbox is created for ms_synthesis_dataset
#                                (dlib: by dlib library, landmark: use ranges of landmark labels, 
#                                   mixed: use dlib when possible, then landmark)
#     "size": "2000",       <-- size of training dataset (inlcuding validation)
#     "ft_ratio": "0.8",      <-- ratio of synthesis-2-real images used in training set
#     "train_test_split": "1.0",    <-- train test split, should be 1.0. Testing is not included in this script

#     "image_dim": "128",       <-- Preprocessing parameters: please check original github project
#     "brightness": "0.24",
#     "saturation": "0.3",
#     "contrast": "0.15",
#     "hue": "0.14",
#     "angle": "14",
#     "face_offset": "32",
#     "crop_offset": "16",
#     "batch_size": "32",

#     "epochs": "80"        <-- number of training epochs
# }
######################################
######################################


# read train.json config 
def readConfig(jsonpath):
    with open(jsonpath, 'r') as f:
        return json.load(f)
    
# write epoch tracking file
def write_history(fn, content):
    with open(fn, 'a+') as f:
        f.write(f'{content}\n')

# wrapper function
def train(foldername, resume=False):

    ###############################
    ###### Training Config #######
    ##############################

    # Load training configuration
    root_dir = os.path.join(os.getcwd(), foldername)
    config = readConfig(os.path.join(root_dir, 'train.json'))
    # timestamp = datetime.datetime.now().strftime('%m%d%H%m')
    print(f'Executing dir: {root_dir}')

    ##########################################
    ############### State saving #############
    ##########################################

    # Load tracking and checkpoint files if provided
    checkpoint_fn = os.path.join(root_dir, f'model.pt')
    epoch_fn = os.path.join(root_dir, f'model.epoch')
    pickle_fn = os.path.join(root_dir, 'data_list.pickle')

    #######################################
    ####### Data Loading #################
    ######################################

    # Parse dataset XML files to create file list (dict)
    xml_dataset = []
    if resume:
        # because list is created randomly, we store pickle file to retain same dataset
        with open(pickle_fn, 'rb') as f:
            xml_dataset = pickle.load(f)
    else:
        xml_dataset = generate_train_data(config["data_dir"], config["bbox_option"],
                                        int(config["size"]), float(config["ft_ratio"]))
        # TODO: make a backup if existing
        with open(pickle_fn, 'wb') as f:
            pickle.dump(xml_dataset, f, pickle.HIGHEST_PROTOCOL)

    # Create dataset object
    print('\nPrepare data')
    train_test_split = float(config["train_test_split"])
    preprocessor = Preprocessor(
        image_dim = int(config["image_dim"]),
        brightness = float(config["brightness"]),
        saturation = float(config["saturation"]),
        contrast = float(config["contrast"]),
        hue = float(config["hue"]),
        angle = int(config["angle"]),
        face_offset = int(config["face_offset"]),
        crop_offset = int(config["crop_offset"]))

    train_images = LandmarksDataset(xml_dataset, preprocessor, train_test_split=train_test_split, train = True)
    ## ! Test is not part of this workflow
    # test_images = LandmarksDataset(xml_dataset, preprocessor, train_test_split=train_test_split, train = False)

    len_val_set = int(0.1 * len(train_images))
    len_train_set = len(train_images) - len_val_set

    print(f'Train: {len_train_set}')
    print(f'Validate: {len_val_set}')
    # print(f'Test: {len(test_images)}')

    train_images, val_images = random_split(train_images, [len_train_set, len_val_set])

    # Create DataLoader
    batch_size = int(config["batch_size"]) or 32
    train_data = torch.utils.data.DataLoader(train_images, batch_size = batch_size, shuffle = True)
    val_data = torch.utils.data.DataLoader(val_images, batch_size = 2 * batch_size, shuffle = False)
    # test_data = torch.utils.data.DataLoader(test_images, batch_size = 2 * batch_size, shuffle = False)

    ###########################################
    ################ Training #################
    ###########################################

    # Create Model
    print('\nInit Model')
    model = XceptionNet()
    model.cuda()
    objective = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.0008)
    start_epoch = 0
    best_loss = np.inf

    # load checkpoint if resume job
    if resume:
        start_epoch, best_loss, model, optimizer = load_ckpt(checkpoint_fn, model, optimizer)

    # Start training
    print('\nTraining')

    # define validate function
    @torch.no_grad()
    def validate(save = None):
        cum_loss = 0.0
        model.eval()
        for features, labels in tqdm(val_data, desc = 'Validating', ncols = 600):
            features = features.cuda()
            labels = labels.cuda()
            outputs = model(features)
            loss = objective(outputs, labels)
            cum_loss += loss.item()
            break
        return cum_loss/len(val_data)

    # Start training really
    epochs = int(config["epochs"]) or 100 # default to 100 epochs
    batches = len(train_data)

    for epoch in range(start_epoch, epochs):
        cum_loss = 0.0
        #################
        ## Train Model ##
        #################
        model.train()
        for batch_idx, (features, labels) in enumerate(tqdm(train_data, desc = f'Epoch({epoch + 1}/{epochs})', ncols = 800)):
            features = features.cuda()
            labels = labels.cuda()

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forwarwd pass
            outputs = model(features)
            # calculate batch loss
            loss = objective(outputs, labels)
            # backward pass
            loss.backward()
            # step
            optimizer.step()
            # loss
            cum_loss += loss.item()

        train_loss = cum_loss / batches
        val_image_name = f'epoch({str(epoch + 1).zfill(len(str(epochs)))}).jpg'
        val_loss = validate(os.path.join('progress', val_image_name))
        if val_loss < best_loss:
            best_loss = val_loss

        # write checkpoint and tracking
        print('Saving epoch....................')
        ckpt = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }
        save_ckpt(checkpoint_fn, ckpt)
        msg = (f'Epoch({epoch + 1}/{epochs}) -> Training Loss: {train_loss:.8f} '
               + f'| Validation Loss: {val_loss:.8f}')
        write_history(epoch_fn, msg)
        print(msg)

def save_ckpt(fn, state):
    torch.save(state, fn)

def load_ckpt(fn, model, optimizer):
    ckpt = torch.load(fn)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"], ckpt["best_loss"], model, optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--folder', required=True)
    parser.add_argument('-r','--resume', action='store_true')

    args = parser.parse_args()
    train(args.folder, args.resume)

    
