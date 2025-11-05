import os
import argparse
import yaml
import random
import math

import numpy as np


import torch
torch.set_num_threads(10)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import wandb
from model import CNNModel, ResNet, BasicBlock
from utils import *


METHODS = ['baseline']


def set_seed(seed=42):
    """
    Set random seeds for all possible random processes.
    :param seed: int
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(model, train_loaders, val_loaders, test_loaders_global, wandb, args, device):

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,  T_max=args.epochs, eta_min=0.000001)
    criterion = nn.CrossEntropyLoss()

    # lists to keep track of losses and accuracies
    train_loss, train_acc = [], []
    valid_loss_p, valid_loss_n = [], []
    valid_acc_p,  valid_acc_n = [], []

    for epoch in range(args.epochs):
        print(f"[INFO]: Epoch {epoch+1} of {args.epochs}")
        
        train_epoch_loss, train_epoch_acc = train(model, train_loaders, 
                                                optimizer, criterion, device)
        
        valid_epoch_loss_p, valid_epoch_acc_p, valid_epoch_loss_n, valid_epoch_acc_n = validate(model, val_loaders[0][0], val_loaders[0][1],  
                                                            criterion, device)
        
        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)

        valid_loss_p.append(valid_epoch_loss_p)
        valid_loss_n.append(valid_epoch_loss_n)

        valid_acc_p.append(valid_epoch_acc_p)
        valid_acc_n.append(valid_epoch_acc_n)

        valid_epoch_loss = (valid_epoch_loss_n + valid_epoch_loss_p)/2
        valid_loss =  valid_loss_p + valid_loss_p
        valid_epoch_acc = (valid_epoch_acc_p + valid_epoch_acc_n)/2

        eval_dict = {f"train_epoch_loss": train_epoch_loss, f"train_epoch_acc": train_epoch_acc,
                     f"valid_epoch_loss": valid_epoch_loss, f"valid_epoch_acc": valid_epoch_acc}

        if wandb:
            wandb.log(eval_dict)

        rtpt.step()
        scheduler.step()

    gt_acc, gt_acc_p, gt_acc_n = test(model, test_loaders_global, device)    

    return  gt_acc_p, gt_acc_n, gt_acc


def train(model, trainloader, optimizer, criterion, device=None):

    model.train()
    print('Training')

    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0

    for _, data in enumerate(trainloader):
        counter += 1

        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # forward pass
        outputs = model(image)

        # calculate the loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()

        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()

        # backpropagation
        loss.backward()

        # update the optimizer parameters
        optimizer.step()
    
    # loss and accuracy for the complete epoch
    loss = train_running_loss / counter
    acc = 100. * (train_running_correct / len(trainloader.dataset))
    return loss, acc

# validation
def validate(model, testloader_p, testloader_n, criterion, device):
    model.eval()
    print('Validation')
    valid_running_loss_p = 0.0
    valid_running_correct_p = 0

    valid_running_loss_n = 0.0
    valid_running_correct_n = 0

    counter_p = 0
    counter_n = 0

    with torch.no_grad():
        for i, data in enumerate(testloader_p):
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(image)

            # calculate the loss
            loss = criterion(outputs, labels)
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            
            counter_p += 1
            valid_running_loss_p += loss.item()
            valid_running_correct_p += (preds == labels).sum().item()


        for i, data in enumerate(testloader_n):
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(image)

            # calculate the loss
            loss = criterion(outputs, labels)
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)

            counter_n += 1
            valid_running_loss_n += loss.item()
            valid_running_correct_n += (preds == labels).sum().item()
        
    # loss and accuracy for the complete epoch
    epoch_loss_p = valid_running_loss_p / counter_p
    epoch_loss_n = valid_running_loss_n / counter_n

    epoch_acc_p = 100. * (valid_running_correct_p / len(testloader_p.dataset))
    epoch_acc_n = 100. * (valid_running_correct_n / len(testloader_n.dataset))

    return epoch_loss_p, epoch_acc_p, epoch_loss_n, epoch_acc_n


def test(model, test_loader, device):

    M_a, M_a_p, M_a_n = [], [], []
    model.eval()
    
    for _, (y_p, y_n) in test_loader.items():
        testloader_p = y_p
        testloader_n = y_n
        correct_p = 0
        correct_n = 0

        with torch.no_grad():
            for i, data in enumerate(testloader_p):
                
                image, labels = data
                image = image.to(device)
                labels = labels.to(device)

                output = model(image)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct_p += pred.eq(labels.view_as(pred)).sum().item()

            for i, data in enumerate(testloader_n):
                
                image, labels = data
                image = image.to(device)
                labels = labels.to(device)

                output = model(image)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct_n += pred.eq(labels.view_as(pred)).sum().item()

        acc_p = 100. * correct_p / len(testloader_p.dataset)
        acc_n = 100. * correct_n / len(testloader_n.dataset)
        
        M_a_p.append(acc_p)
        M_a_n.append(acc_n)
        M_a.append((acc_n+acc_p)/2)

    return M_a, M_a_p, M_a_n


def get_dataset(args):

    # the training transforms
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    
    train_t0 = args.train_path_global

    val_t0 = args.val_path_global

    test_global_path = args.test_path_global

    BATCH_SIZE = args.batch_size
    kwargs = {'num_workers':0, 'pin_memory':False} 

    test_global_dataset = datasets.ImageFolder(root=test_global_path, transform=transformer)
    positives = [i for i, (x, y) in enumerate(test_global_dataset) if  y == 1.0]
    negatives = [i for i, (x, y) in enumerate(test_global_dataset) if  y == 0.0]
    testdataset_p = torch.utils.data.Subset(test_global_dataset, positives)
    testdataset_n = torch.utils.data.Subset(test_global_dataset, negatives)
    test_loader_p = DataLoader(dataset=testdataset_p, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    test_loader_n = DataLoader(dataset=testdataset_n, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    test_loaders = {0: [test_loader_p, test_loader_n]}
    
    count = 1
    # data_names = ['case_disjoint', 'case_strict']
    # data_names = ['global_pix_100']
    # for data in data_names:

    #     test_t0 = f'{args.results_dir}/{data}/test/images/t0'
    #     test_t1 = f'{args.results_dir}/{data}/test/images/t1'
    #     test_t2 = f'{args.results_dir}/{data}/test/images/t2'

    #     testdataset0 = datasets.ImageFolder(root=test_t0, transform=transformer)
    #     positives = [i for i, (x, y) in enumerate(testdataset0) if  y == 1.0]
    #     negatives = [i for i, (x, y) in enumerate(testdataset0) if  y == 0.0]
    #     testdataset0_p = torch.utils.data.Subset(testdataset0, positives)
    #     testdataset0_n = torch.utils.data.Subset(testdataset0, negatives)

    #     testdataset1 = datasets.ImageFolder(root=test_t1, transform=transformer)
    #     positives = [i for i, (x, y) in enumerate(testdataset1) if  y == 1.0]
    #     negatives = [i for i, (x, y) in enumerate(testdataset1) if  y == 0.0]
    #     testdataset1_p = torch.utils.data.Subset(testdataset1, positives)
    #     testdataset1_n = torch.utils.data.Subset(testdataset1, negatives)

    #     testdataset2 = datasets.ImageFolder(root=test_t2, transform=transformer)
    #     positives = [i for i, (x, y) in enumerate(testdataset2) if  y == 1.0]
    #     negatives = [i for i, (x, y) in enumerate(testdataset2) if  y == 0.0]
    #     testdataset2_p = torch.utils.data.Subset(testdataset2, positives)
    #     testdataset2_n = torch.utils.data.Subset(testdataset2, negatives)

    #     test0_loader_p = DataLoader(dataset= testdataset0_p, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    #     test1_loader_p = DataLoader(dataset= testdataset1_p, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    #     test2_loader_p = DataLoader(dataset= testdataset2_p, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    #     test0_loader_n = DataLoader(dataset= testdataset0_n, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    #     test1_loader_n = DataLoader(dataset= testdataset1_n, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    #     test2_loader_n = DataLoader(dataset= testdataset2_n, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    #     test_loaders[count] = [test0_loader_p, test0_loader_n]
    #     count+=1
    #     test_loaders[count] = [test1_loader_p, test1_loader_n]
    #     count+=1
    #     test_loaders[count] = [test2_loader_p, test2_loader_n]
    #     count+=1
    
    traindataset0 = datasets.ImageFolder(root=train_t0, transform=transformer)
    valdataset0 = datasets.ImageFolder(root=val_t0, transform=transformer)
    positives = [i for i, (x, y) in enumerate(valdataset0) if  y == 1.0]
    negatives = [i for i, (x, y) in enumerate(valdataset0) if  y == 0.0]
    valdataset0_p = torch.utils.data.Subset(valdataset0, positives)
    valdataset0_n = torch.utils.data.Subset(valdataset0, negatives)

    val_loader_p = DataLoader(dataset= valdataset0_p, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    val_loader_n = DataLoader(dataset= valdataset0_n, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    train_loader_g = DataLoader(dataset= traindataset0, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    
    val_loaders_g = {0: [val_loader_p, val_loader_n]}



    return train_loader_g, val_loaders_g , test_loaders



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Continual Confounding Dataset')

    parser.add_argument('--wandb', action='store_true', help='Log run to Weights and Biases.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rtpt', type=str, help='Use rtpt to set process name with given initials.', required=True)
    parser.add_argument('--method', type=str, required=True, choices=METHODS)
    parser.add_argument('-results', '--results_dir', type=str, help='path to store results', required=True)
    parser.add_argument('-name', '--dataset_type', type=str, help='name of experiment')
    parser.add_argument('--save_model', action='store_true', help='save model weights', default=True)
    parser.add_argument('-e', '--epochs', type=int, default=50, help='number of epochs to train our network for')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='number of images per batch')
    parser.add_argument('-m', '--model_name', type=str, default='Resnet', help='select the model you would like to run training with')

    parser.add_argument('-gt0', '--test_path_global', type=str, help='path for global test')
    parser.add_argument('-gtr0', '--train_path_global', type=str, help='path for global train')
    parser.add_argument('-gtv', '--val_path_global', type=str, help='path for global val')

    args = parser.parse_args()

    seed = args.seed
    set_seed(args.seed)

    run_id =  wandb.util.generate_id()
    print(f'Sampled new wandb run_id {run_id}.')
    
    wandb.init(project='confounding_continual_dataset', name=os.path.dirname(args.dataset_type),
                id=run_id, resume=True)

    if args.rtpt is not None:
        from rtpt import RTPT
        rtpt = RTPT(name_initials=args.rtpt, experiment_name=f'{args.dataset_type}_{args.method}', max_iterations=args.epochs)
        rtpt.start()

    out_dir = os.path.dirname(args.results_dir)
    save_model = args.save_model
    dataset_type = f"{args.dataset_type}/{args.method}"
    model_name = args.model_name
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    if model_name == 'Resnet':
        model = ResNet(img_channels=3, num_layers=18, num_classes=2, block=BasicBlock).to(device)
        model_name = 'Resnet18'
    

    
    
    print('Loading datasets...')
    train_loaders, val_loaders, test_loaders = get_dataset(args)
    
    test_acc_p, test_acc_n, test_acc = run(model, train_loaders, val_loaders, test_loaders, wandb, args, device)

    results = args.results_dir
    if not os.path.isdir(f'{results}/{dataset_type}/{model_name}/seed_{seed}'):
        os.makedirs(f'{results}/{dataset_type}/{model_name}/seed_{seed}')

    np.save(f'{results}/{dataset_type}/{model_name}/seed_{seed}/test_acc_p.npy', test_acc_p)
    np.save(f'{results}/{dataset_type}/{model_name}/seed_{seed}/test_acc_n.npy', test_acc_n)
    np.save(f'{results}/{dataset_type}/{model_name}/seed_{seed}/test_acc.npy', test_acc)
    
    print(test_acc)
    print('--------positive------')
    print(test_acc_p)
    print('--------negative------')
    print(test_acc_n)

