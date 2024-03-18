# NATASHA'S CODE FOR BASIC TRAINING OF BASIC NEURAL NETWORKS
# INCLUDES THE FUNCTIONS I'VE WRITTEN FOR CROSS VALIDATION, CLASSIFICATION AND REGRESSION

import torch
from torch import nn
from torcheval.metrics import R2Score
from torch.utils.data import DataLoader
# import model_architectures as ma

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
import seaborn as sns

import pandas as pd
import numpy as np
import math
import itertools
from datetime import datetime
import os
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

###############################################################
# CROSS VALIDATION
###############################################################
def optimise_hyperparameters(X_data, 
                             y_data, 
                             params,
                             file_path,
                             normalise=True,
                             ):
    
    # find all combinations of parameters to test
    cv_sets = list(itertools.product(*params.values()))
    
    # test one set of parameters

    for cv_set in tqdm(cv_sets):
        # set the parameters as variables
        torch.manual_seed(47)
        model_name = cv_set[list(params.keys()).index("model_name")]
        lr = cv_set[list(params.keys()).index("lr")]
        lr_scheduler = cv_set[list(params.keys()).index("lr_scheduler")]
        epochs = cv_set[list(params.keys()).index("epochs")]
        batch_size = cv_set[list(params.keys()).index("batch_size")]
        optimiser = cv_set[list(params.keys()).index("optimiser")]
        weight_decay = cv_set[list(params.keys()).index("weight_decay")]
        run_cross_validation(X_data, 
                             y_data,
                             model_name, 
                             lr, 
                             lr_scheduler, 
                             epochs, 
                             batch_size, 
                             optimiser, 
                             weight_decay,
                             file_path,
                             normalise)

        
def run_cross_validation(X_data,
                         y_data,
                         model_name, 
                         lr, 
                         lr_scheduler, 
                         epochs, 
                         batch_size, 
                         optimiser, 
                         weight_decay, 
                         file_path,
                         normalise=True):

    # define the different k-fold datasets
    kfold = KFold(n_splits=5, 
                  shuffle=True, 
                  random_state=47)
    kfold_indices = [*kfold.split(X_data)]

    # define whether this is a multi-class problem
    isMulti = len(np.unique(y_data)) > 2 
    
    # save all of the the metrics for every fold
    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    
    # create each k-fold dataset
    for train_indices, test_indices in kfold_indices:
        # create the dataset
        X_train = X_data[train_indices]
        y_train = y_data[train_indices]
        X_test = X_data[test_indices]
        y_test = y_data[test_indices]

        # process the data into the correct format
        train_loader, test_loader = process_data(X_train, 
                                                 X_test, 
                                                 y_train, 
                                                 y_test, 
                                                 batch_size=batch_size,
                                                 normalise=True)
        
        # initialise a new model
        model = ma.create_model(model_name)
        # train the model using each k-fold dataset
        train_loss, test_loss, train_acc, test_acc = train(train_loader,
                                                           test_loader,
                                                           model,
                                                           lr, 
                                                           lr_scheduler, 
                                                           epochs, 
                                                           batch_size, 
                                                           optimiser, 
                                                           weight_decay,
                                                           isMulti)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
    # save the results, including the average from the 5-folds
    results = pd.DataFrame({"date": [datetime.now()], 
               "model": [model_name], 
               "lr": [lr], 
               "lr_scheduler": [lr_scheduler], 
               "epochs": [epochs], 
               "batch_size": [batch_size], 
               "optimiser": [optimiser], 
               "weight_decay": [weight_decay],
               "all_train_losses": [train_losses],
               "all_train_accs": [train_accs],
               "all_test_losses": [test_losses],
               "all_test_accs": [test_accs],
               "avg_train_loss": [np.mean(train_losses)],
               "avg_train_acc": [np.mean(train_accs)],
               "avg_test_loss": [np.mean(test_losses)],
               "avg_test_acc": [np.mean(test_accs)],
                })
    if os.path.isfile(file_path): 
        results.to_csv(file_path, mode="a", index=False, header=False)
    else:
        results.to_csv(file_path, mode="a", index=False, header=True)

        
def process_data(X_train=None, 
                 X_test=None, 
                 y_train=None, 
                 y_test=None, 
                 batch_size=32, 
                 normalise=True,
                 use_test=True,
                 ):
    ## I know this code is horrible, I need to fix it, I'm just lazy
    
    # normalise the data
    if normalise:
        normaliser = StandardScaler().fit(X_train)
        X_train = normaliser.transform(X_train)
        if use_test:
            X_test = normaliser.transform(X_test)

    # convert the data to tensors, give the correct data type and shape
    X_train = torch.from_numpy(X_train).type(torch.float32)
    if use_test:
        X_test = torch.from_numpy(X_test).type(torch.float32)
    isMulti = len(np.unique(y_train)) > 2
    if isMulti:
        y_train = torch.from_numpy(y_train).type(torch.int64)
        if use_test:
            y_test = torch.from_numpy(y_test).type(torch.int64)
    else:
        y_train = torch.from_numpy(y_train).type(torch.float32).unsqueeze(dim=1)
        if use_test:
            y_test = torch.from_numpy(y_test).type(torch.float32).unsqueeze(dim=1)

    # put the data into dataloaders
    torch.manual_seed(47)
    train_loader = DataLoader(dataset=list(zip(X_train, y_train)),
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                             )
    
    torch.manual_seed(47)
    if use_test:
        test_loader = DataLoader(dataset=list(zip(X_test, y_test)),
                                 batch_size=batch_size,
                                 shuffle=True,
                                 pin_memory=True,
                                )
    if use_test:
        return train_loader, test_loader
    else:
        return train_loader

def train(train_loader,
          test_loader,
          model,
          lr, 
          lr_scheduler, 
          epochs, 
          batch_size, 
          optimiser, 
          weight_decay,
          isMulti):
    
    # put the model on the device
    model = model.to(device)
    # is the problem is multi-class or binary classes
    if isMulti:
        loss_func = nn.CrossEntropyLoss() # cross entropy for multi-class
    else:
        loss_func = nn.BCEWithLogitsLoss() # binary cross entropy for binary
    
    # define the optimiser
    optim_dict = {"SGD": torch.optim.SGD(params=model.parameters(), 
                                         lr=lr, 
                                         weight_decay=weight_decay),
                  "Adam": torch.optim.Adam(params=model.parameters(), 
                                           lr=lr, 
                                           weight_decay=weight_decay),
                 }
    optimiser = optim_dict[optimiser]
    
    # define the learning rate scheduler
    if lr_scheduler:
        scheduler_dict = {"OneCycleLR": torch.optim.lr_scheduler.OneCycleLR, # max_lr (upper boundary)
                          "StepLR": torch.optim.lr_scheduler.StepLR, # step_size (lr decays every x steps)
                          "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR, # T_max (num steps per half period of cos)
                          "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts # T_0 (num steps per half period of cos)
                 }
        if lr_scheduler[0] == "OneCycleLR":
            lr_scheduler[1]["epochs"] = epochs
            lr_scheduler[1]["steps_per_epoch"] = len(train_loader)
        lr_scheduler = scheduler_dict[lr_scheduler[0]](optimizer=optimiser, **lr_scheduler[1])
        
    # training loop
    for epoch in range(1, epochs+1):
        
        # training
        model.train()
            
        # for each mini-batch
        for X_mini_train, y_mini_train in train_loader:
            
            X_mini_train = X_mini_train.to(device)
            y_mini_train = y_mini_train.to(device)
            
            y_mini_train_pred = model(X_mini_train)
            train_loss = loss_func(y_mini_train_pred, y_mini_train)
            optimiser.zero_grad() 
            train_loss.backward() 
            optimiser.step()
            if lr_scheduler:
                lr_scheduler.step()
        
    model.eval()
    with torch.no_grad():
                
        running_train_loss = 0
        running_train_correct = 0
        running_test_loss = 0
        running_test_correct = 0
                
        # find loss and accuracy for each minibatch in train 
        for X_mini_train, y_mini_train in train_loader:
            X_mini_train = X_mini_train.to(device)
            y_mini_train = y_mini_train.to(device)
            # calculate loss
            y_mini_train_pred = model(X_mini_train)
            train_loss = loss_func(y_mini_train_pred, y_mini_train)
            running_train_loss += train_loss.item()
            # calculate accuracy
            if isMulti:
                y_mini_train_pred = torch.softmax(y_mini_train_pred, dim=1).argmax(dim=1)
            else:
                y_mini_train_pred = torch.round(torch.sigmoid(y_mini_train_pred))
            running_train_correct += sum(y_mini_train_pred == y_mini_train).item()

        # find loss and accuracy for each minibatch in test
        for X_mini_test, y_mini_test in test_loader:
            X_mini_test = X_mini_test.to(device)
            y_mini_test = y_mini_test.to(device)
            # calculate loss
            y_mini_test_pred = model(X_mini_test)
            test_loss = loss_func(y_mini_test_pred, y_mini_test)
            running_test_loss += test_loss.item()
            # calculate accuracy
            if isMulti:
                y_mini_test_pred = torch.softmax(y_mini_test_pred, dim=1).argmax(dim=1)
            else:
                y_mini_test_pred = torch.round(torch.sigmoid(y_mini_test_pred))
            running_test_correct += sum(y_mini_test_pred == y_mini_test).item()

        # find the loss and accuracy for the full dataset
        train_loss = running_train_loss / len(train_loader) # divide by number of batches
        test_loss = running_test_loss / len(test_loader)
        train_acc = (running_train_correct / len(train_loader.dataset)) * 100 # divide by number of datapoints
        test_acc = (running_test_correct / len(test_loader.dataset)) * 100
        
        return train_loss, test_loss, train_acc, test_acc
    
###################################################################
# CLASSIFICATION
###################################################################

def train_classification(model, 
                         X_data: np.ndarray, 
                         y_data: np.ndarray,
                         batch_size=32,
                         optimiser="SGD", 
                         lr=0.01, 
                         lr_scheduler=None,
                         epochs=1000, 
                         normalise=True,
                         weight_decay=0):
    
    # if X data only has one feature, add an empty 2nd dimension
    if len(X_data.shape) == 1:
        X_data = np.expand_dims(X_data, axis=1)
        
    # create a train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_data, 
                                                        y_data, 
                                                        test_size=0.2,
                                                        random_state=42)
    
    # process the data and store in dataloaders
    train_loader, test_loader = process_data(X_train, X_test, y_train, y_test, batch_size, normalise)

    # store the losses for plotting later
    all_epochs_losses = []
    all_full_train_losses = []
    all_full_test_losses = []
    
    # put the model on the device
    model = model.to(device)
    
    # define whether this is a multi-class problem
    isMulti = len(np.unique(y_train)) > 2 
    
    # define the loss function
    if isMulti:
        loss_func = nn.CrossEntropyLoss() # cross entropy for multi-class
    else:
        loss_func = nn.BCEWithLogitsLoss() # binary cross entropy for binary
    
    # define the optimiser
    optim_dict = {"SGD": torch.optim.SGD(params=model.parameters(), lr=lr, weight_decay=weight_decay),
                  "Adam": torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)}
    optimiser = optim_dict[optimiser]
    
    # define the learning rate scheduler
    if lr_scheduler:
        scheduler_dict = {"OneCycleLR": torch.optim.lr_scheduler.OneCycleLR, # max_lr (upper boundary)
                          "StepLR": torch.optim.lr_scheduler.StepLR, # step_size (lr decays every x steps)
                          "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR, # T_max (num steps per half period of cos)
                          "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts # T_0 (num steps per half period of cos)
                 }
        if lr_scheduler[0] == "OneCycleLR":
            lr_scheduler[1]["epochs"] = epochs
            lr_scheduler[1]["steps_per_epoch"] = len(train_loader)
        lr_scheduler = scheduler_dict[lr_scheduler[0]](optimizer=optimiser, **lr_scheduler[1])

    # training loop
    for epoch in range(1, epochs+1):
        
        # for each mini-batch
        for X_mini_train, y_mini_train in train_loader:
            
            # training
            model.train()
            
            X_mini_train = X_mini_train.to(device)
            y_mini_train = y_mini_train.to(device)
            
            y_mini_train_pred = model(X_mini_train)
            train_loss = loss_func(y_mini_train_pred, y_mini_train)
            optimiser.zero_grad() 
            train_loss.backward() 
            optimiser.step()
            if lr_scheduler:
                lr_scheduler.step()
        
        # evaluation
        if (epoch % math.ceil(epochs/20) == 0) or (epoch == 1):
            
            model.eval()
            with torch.no_grad():
                
                running_train_loss = 0
                running_train_correct = 0
                running_test_loss = 0
                running_test_correct = 0
                
                # find loss and accuracy for each minibatch in train 
                for X_mini_train, y_mini_train in train_loader:
                        
                    X_mini_train = X_mini_train.to(device)
                    y_mini_train = y_mini_train.to(device)
                    
                    # calculate loss
                    y_mini_train_pred = model(X_mini_train)
                    train_loss = loss_func(y_mini_train_pred, y_mini_train)
                    running_train_loss += train_loss 
                    
                    # calculate accuracy
                    if isMulti:
                        y_mini_train_pred = torch.softmax(y_mini_train_pred, dim=1).argmax(dim=1)
                    else:
                        y_mini_train_pred = torch.round(torch.sigmoid(y_mini_train_pred))
                    running_train_correct += sum(y_mini_train_pred == y_mini_train).item()
                
                # find loss and accuracy for each minibatch in test
                for X_mini_test, y_mini_test in test_loader:
                        
                    X_mini_test = X_mini_test.to(device)
                    y_mini_test = y_mini_test.to(device)
                    
                    # calculate loss
                    y_mini_test_pred = model(X_mini_test)
                    test_loss = loss_func(y_mini_test_pred, y_mini_test)
                    running_test_loss += test_loss 
                    
                    # calculate accuracy
                    if isMulti:
                        y_mini_test_pred = torch.softmax(y_mini_test_pred, dim=1).argmax(dim=1)
                    else:
                        y_mini_test_pred = torch.round(torch.sigmoid(y_mini_test_pred))
                    running_test_correct += sum(y_mini_test_pred == y_mini_test).item()
                
                # find the loss and accuracy for the full dataset
                full_train_loss = running_train_loss / len(train_loader) # divide by number of batches
                full_test_loss = running_test_loss / len(test_loader)
                full_train_acc = (running_train_correct / len(train_loader.dataset)) * 100 # divide by number of datapoints
                full_test_acc = (running_test_correct / len(test_loader.dataset)) * 100
                all_epochs_losses.append(epoch)
                all_full_train_losses.append(full_train_loss.item())
                all_full_test_losses.append(full_test_loss.item())
        
                # print the losses and accuracies
                print(f"Epoch: {epoch} | ", 
                      f"Train loss: {full_train_loss:.3f} | ", 
                      f"Test loss: {full_test_loss:.3f} | ", 
                      f"Train accuracy: {full_train_acc:.2f}% | ", 
                      f"Test accuracy: {full_test_acc:.2f}%")
                
    # plot the losses over time
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xticks(all_epochs_losses[::2])
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.set_xlabel("Epochs", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.plot(all_epochs_losses, 
            all_full_train_losses, 
            c="firebrick", 
            label="Training Loss",
            marker="o",
            linewidth=2,)
    ax.plot(all_epochs_losses,
            all_full_test_losses, 
            c="dodgerblue", 
            label="Test Loss",
            marker="o",
            linewidth=2,)
    plt.legend(fontsize=14);
    
    return model

def predict_classification(opt_model,
                           X_data,
                           y_data):
    
    if len(X_data.shape) == 1:
        X_data = np.expand_dims(X_data, axis=1)
        
    opt_model = opt_model.to(device)
    
    num_classes = len(np.unique(y_data))
    isMulti = num_classes > 2

    # convert data to tensors
    X_data = torch.from_numpy(X_data).type(torch.float32).to(device)
    if isMulti:
        y_data = torch.from_numpy(y_data).type(torch.int64).to(device)
    else:
        y_data = torch.from_numpy(y_data).type(torch.float32).unsqueeze(dim=1).to(device)
        
    opt_model.eval()
    with torch.no_grad():
        # forward pass through the model
        y_pred = opt_model(X_data)

        # predict the classes
        if isMulti:
            y_pred_class = torch.softmax(y_pred, dim=1).argmax(dim=1)
        else:
            y_pred_class = torch.round(torch.sigmoid(y_pred))

        # print accuracy
        print(f"Accuracy: {sum(y_pred_class == y_data).item() / len(y_data) * 100:.2f}%")

        # plot a confusion matrix
        cm = confusion_matrix(y_data.detach().cpu(), y_pred_class.detach().cpu())
        cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
        fig, ax = plt.subplots(figsize=(num_classes,num_classes))
        ax = sns.heatmap(cmn, 
                        annot=True, 
                        fmt=".2f",
                        norm=SymLogNorm(linthresh=3),
                        xticklabels=list(range(0, num_classes)), 
                        yticklabels=list(range(0, num_classes)))
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        for t in ax.texts: t.set_text(t.get_text() + "%")
        
        # if the data is 2d, plot the class predictions               
        if X_data.shape[1] == 2:
            x_min, x_max = X_data[:, 0].min(), X_data[:, 0].max()
            y_min, y_max = X_data[:, 1].min(), X_data[:, 1].max()
            xx, yy = np.meshgrid(np.linspace(x_min.detach().cpu(), x_max.detach().cpu(), 500), 
                                 np.linspace(y_min.detach().cpu(), y_max.detach().cpu(), 500))
            mesh_logits = opt_model(torch.from_numpy(np.column_stack((xx.ravel(),
                                                                  yy.ravel()))).to(device).float())
            if isMulti:
                mesh_pred = torch.softmax(mesh_logits, dim=1).argmax(dim=1)
            else:
                mesh_pred = torch.round(torch.sigmoid(mesh_logits))
            mesh_pred = mesh_pred.reshape(xx.shape).detach().cpu().numpy()
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pcolormesh(xx, yy, mesh_pred, cmap=plt.cm.tab10, alpha=0.7)
            ax.scatter(X_data.cpu()[:, 0], X_data.cpu()[:, 1], 
                       c=y_data.cpu(), cmap=plt.cm.tab10,
                       s=40)
            ax.set_xlim(xx.min(), xx.max()), ax.set_ylim(yy.min(), yy.max())

def train_final_model(model,
                      X_data: np.ndarray, 
                      y_data: np.ndarray,
                      batch_size=32, 
                      optimiser="SGD", 
                      lr=0.01, 
                      lr_scheduler=None, 
                      epochs=100, 
                      normalise=True,
                      weight_decay=0):

    # if the X data has only one feature, add an empty second dimension
    if len(X_data.shape) == 1:
        X_data = np.expand_dims(X_data, axis=1)

    # process the data and store in dataloaders
    data_loader = process_data(X_train=X_data, y_train=y_data, use_test=False, normalise=normalise, batch_size=batch_size)
    
    # put the model on the device
    model = model.to(device)
    
    # define whether this is a multi-class problem
    isMulti = len(np.unique(y_data)) > 2 
    
    # define the loss function
    if isMulti:
        loss_func = nn.CrossEntropyLoss() # cross entropy for multi-class
    else:
        loss_func = nn.BCEWithLogitsLoss() # binary cross entropy for binary
        
    # define the optimiser
    optim_dict = {"SGD": torch.optim.SGD(params=model.parameters(), lr=lr, weight_decay=weight_decay),
                  "Adam": torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)}
    optimiser = optim_dict[optimiser]

    # define the learning rate scheduler
    if lr_scheduler:
        scheduler_dict = {"OneCycleLR": torch.optim.lr_scheduler.OneCycleLR, # max_lr (upper boundary)
                          "StepLR": torch.optim.lr_scheduler.StepLR, # step_size (lr decays every x steps)
                          "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR, # T_max (num steps per half period of cos)
                          "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts # T_0 (num steps per half period of cos)
                 }
        if lr_scheduler[0] == "OneCycleLR":
            lr_scheduler[1]["epochs"] = epochs
            lr_scheduler[1]["steps_per_epoch"] = len(data_loader)
        lr_scheduler = scheduler_dict[lr_scheduler[0]](optimizer=optimiser, **lr_scheduler[1])
    
    # training loop
    for epoch in range(1, epochs+1):
        
        # for each mini-batch
        for X_mini_data, y_mini_data in data_loader:
            
            # training
            model.train()
            
            X_mini_data = X_mini_data.to(device)
            y_mini_data = y_mini_data.to(device)
            
            y_mini_data_pred = model(X_mini_data)
            data_loss = loss_func(y_mini_data_pred, y_mini_data)
            optimiser.zero_grad() 
            data_loss.backward() 
            optimiser.step()
            if lr_scheduler:
                lr_scheduler.step()
                
    model.eval()
    with torch.no_grad():

        running_data_loss = 0
        running_data_correct = 0

        # find loss and accuracy for each minibatch in train 
        for X_mini_data, y_mini_data in data_loader:

            X_mini_data = X_mini_data.to(device)
            y_mini_data = y_mini_data.to(device)

            # calculate loss
            y_mini_data_pred = model(X_mini_data)
            data_loss = loss_func(y_mini_data_pred, y_mini_data)
            running_data_loss += data_loss 

            # calculate accuracy
            if isMulti:
                y_mini_data_pred = torch.softmax(y_mini_data_pred, dim=1).argmax(dim=1)
            else:
                y_mini_data_pred = torch.round(torch.sigmoid(y_mini_data_pred))
            running_data_correct += sum(y_mini_data_pred == y_mini_data).item()

        # find the loss and accuracy for the full dataset
        full_data_loss = running_data_loss / len(data_loader) # divide by number of batches
        full_data_acc = (running_data_correct / len(data_loader.dataset)) * 100 # divide by number of datapoints
        
        # print the losses and accuracies
        print(f"Final training data loss: {full_data_loss:.3f} | ", 
              f"Final training data accuracy: {full_data_acc:.2f}%")
    return model
####################################################################################################################
# REGRESSION
####################################################################################################################

def train_regression(model, 
                     X_data: np.ndarray, 
                     y_data: np.ndarray, 
                     batch_size=32,
                     loss_choice="MSE", 
                     optim_choice="SGD", 
                     lr=0.01, 
                     epochs=1000, 
                     weight_decay=0):
    
    # if X data only has one feature, add an empty 2nd dimension
    if len(X_data.shape) == 1:
        X_data = np.expand_dims(X_data, axis=1)
        
    # create a train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_data, 
                                                        y_data, 
                                                        test_size=0.2,
                                                        random_state=42)
    
    # normalise the X data, with the scaler based on the training set
    normaliser = StandardScaler().fit(X_train)
    X_train = normaliser.transform(X_train)
    X_test = normaliser.transform(X_test)
    
    # convert the data to tensors, float32 type, give y data shape [samples, 1]
    X_train = torch.from_numpy(X_train).type(torch.float32)
    X_test = torch.from_numpy(X_test).type(torch.float32)
    y_train = torch.from_numpy(y_train).type(torch.float32).unsqueeze(dim=1)
    y_test = torch.from_numpy(y_test).type(torch.float32).unsqueeze(dim=1)
    
    # store the losses for plotting later
    all_full_train_losses = []
    all_full_test_losses = []
    all_epochs_losses = []
    
    # create the model
    torch.manual_seed(seed=42)
    model = model.to(device)
    
    # define the loss and optimisation functions
    loss_dict = {"MSE": nn.MSELoss()}
    optim_dict = {"SGD": torch.optim.SGD(params=model.parameters(), lr=lr, weight_decay=weight_decay),
                  "Adam": torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)}
    loss_func = loss_dict[loss_choice]
    optimizer = optim_dict[optim_choice]
    
    # put the data into dataloaders
    train_loader = DataLoader(dataset=list(zip(X_train, y_train)),
                              batch_size=batch_size,
                              shuffle=True)
                                
    test_loader = DataLoader(dataset=list(zip(X_test, y_test)),
                             batch_size=batch_size,
                             shuffle=True)
    
    # training loop
    for epoch in range(1, epochs+1):
        
        # for each mini-batch
        for X_mini_train, y_mini_train in train_loader:
            
            # training
            model.train()
            
            X_mini_train = X_mini_train.to(device)
            y_mini_train = y_mini_train.to(device)
            
            y_mini_train_pred = model(X_mini_train)
            train_loss = loss_func(y_mini_train_pred, y_mini_train)
            optimizer.zero_grad() 
            train_loss.backward() 
            optimizer.step()
        
        # evaluation
        if (epoch % math.ceil(epochs/20) == 0) or (epoch == 1):
            
            model.eval()
            with torch.no_grad():
                
                running_test_loss = 0
                running_train_loss = 0
                
                # find loss and accuracy for each minibatch in train 
                for X_mini_train, y_mini_train in train_loader:
                        
                    X_mini_train = X_mini_train.to(device)
                    y_mini_train = y_mini_train.to(device)
                    
                    # calculate loss
                    y_mini_train_pred = model(X_mini_train)
                    train_loss = loss_func(y_mini_train_pred, y_mini_train)
                    running_train_loss += train_loss 
                
                # find loss and accuracy for each minibatch in test
                for X_mini_test, y_mini_test in test_loader:
                        
                    X_mini_test = X_mini_test.to(device)
                    y_mini_test = y_mini_test.to(device)
                    
                    # calculate loss
                    y_mini_test_pred = model(X_mini_test)
                    test_loss = loss_func(y_mini_test_pred, y_mini_test)
                    running_test_loss += test_loss 
                
                # find the loss for the full dataset
                full_train_loss = running_train_loss / len(train_loader) # divide by number of batches
                full_test_loss = running_test_loss / len(test_loader)
                all_epochs_losses.append(epoch)
                all_full_train_losses.append(full_train_loss.item())
                all_full_test_losses.append(full_test_loss.item())
        
                # print the losses and accuracies
                print(f"Epoch: {epoch} | ", 
                      f"Train loss: {full_train_loss:.3f} | ", 
                      f"Test loss: {full_test_loss:.3f}")

    # plot the losses over time
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xticks(all_epochs_losses[::2])
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.set_xlabel("Epochs", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.plot(all_epochs_losses, 
            all_full_train_losses, 
            c="firebrick", 
            label="Training Loss",
            marker="o",
            linewidth=2,)
    ax.plot(all_epochs_losses,
            all_full_test_losses, 
            c="dodgerblue", 
            label="Test Loss",
            marker="o",
            linewidth=2,)
    plt.legend(fontsize=14);
    
    return model

def predict_regression(opt_model,
                       X_data,
                       y_data):

    # convert data to tensors
    X_data = torch.from_numpy(X_data).type(torch.float32).to(device)
    y_data = torch.from_numpy(y_data).type(torch.float32).unsqueeze(dim=1).to(device)
    
    opt_model.eval()
    with torch.no_grad():
        # forward pass through the model
        opt_model = opt_model.to(device)
        y_pred = opt_model(X_data)
    
        # find the mse loss
        mse_loss_func = nn.MSELoss()
        mse_loss = mse_loss_func(y_pred, y_data)
        print(f"MSE loss: {mse_loss.item():.3f}")
    
        # if the data is 2d, plot the regression predictions      
        if X_data.shape[1] == 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(X_data.cpu(), y_data.cpu(), c="rebeccapurple", label="Real Data")
            ax.scatter(X_data.cpu(), y_pred.cpu().detach(), c="mediumseagreen", label="Predictions")
            plt.legend(fontsize=14)