import os
import argparse
import random
from tqdm import tqdm
import copy

import numpy as np

import torch
torch.set_num_threads(6)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import wandb
from model import ResNet, BasicBlock
from utils import *

METHODS = ['pnns']


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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model_output(columns, lateral_connections, inputs, task_id):
    features = []
            
    for i in range(task_id):
        feat = columns[i].extract_features(inputs)
        features.append(feat)
    
    current_feat = columns[task_id].extract_features(inputs)
    features.append(current_feat)
    
    if task_id > 0:
        lateral_feats = []
        for i in range(task_id):
            lateral_feat = lateral_connections[task_id-1][i](features[i])
            lateral_feats.append(lateral_feat)
        
        combined_feat = torch.cat([current_feat] + lateral_feats, dim=1)
    else:
        combined_feat = current_feat
    
    outputs = columns[task_id].fc(combined_feat)

    return outputs


def test(columns, lateral_connections, test_loader, current_task_id, device):

    M_a, M_a_p, M_a_n = [], [], []

    for task_id, (y_p, y_n) in test_loader.items():
        
        testloader_p = y_p
        testloader_n = y_n
        correct_p = 0
        correct_n = 0

        if len(test_loader) != 1:
            if task_id < len(columns):
                current_task_id = task_id
            else:
                current_task_id = len(columns) - 1

        print(f"Testing task {task_id} with current task {current_task_id}")

        columns[current_task_id].eval()

        with torch.no_grad():
            for i, data in enumerate(testloader_p):
                
                image, labels = data
                image = image.to(device)
                labels = labels.to(device)

                output = get_model_output(columns, lateral_connections, image, current_task_id)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct_p += pred.eq(labels.view_as(pred)).sum().item()

            for i, data in enumerate(testloader_n):
                
                image, labels = data
                image = image.to(device)
                labels = labels.to(device)

                output = get_model_output(columns, lateral_connections, image, current_task_id)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct_n += pred.eq(labels.view_as(pred)).sum().item()

        acc_p = 100. * correct_p / len(testloader_p.dataset)
        acc_n = 100. * correct_n / len(testloader_n.dataset)
        
        M_a_p.append(acc_p)
        M_a_n.append(acc_n)
        M_a.append((acc_n+acc_p)/2)

    print(M_a)
    return M_a, M_a_p, M_a_n


def classwise_split(targets):
    """
    Returns a dictionary with classwise indices for any class key given labels array.
    Arguments:
        indices (sequence): a sequence of indices
    """
    targets = np.array(targets)
    indices = targets.argsort()
    class_labels_dict = dict()

    for idx in indices:
        if targets[idx] in class_labels_dict: class_labels_dict[targets[idx]].append(idx)
        else: class_labels_dict[targets[idx]] = [idx]

    return class_labels_dict


def get_dataloader(train_dataset,
                       batch_size=None, drop_last=True):

    x_buf, y_buf = list(map(list, zip(*train_dataset)))
    dataset = TensorDataset(torch.stack(x_buf), torch.tensor(y_buf))
    
    kwargs = {'num_workers': 8, 'pin_memory': True}
    sampler = None
    shuffle = True
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=drop_last,
                                                shuffle=shuffle, sampler=sampler, **kwargs)
    
    print(len(train_loader.dataset))
    return train_loader


def run(method, model, train_loaders, val_loaders, test_loaders, test_loaders_global, wandb, args, device):

    t_a_p, t_a_n, tt_acc = [], [], []

    # Initialize first column
    columns = [copy.deepcopy(model).to(device)]
    feature_dim = model.fc.in_features
    
    # Ensure first column has binary output
    columns[0].fc = nn.Linear(feature_dim, 2).to(device)
    
    all_lateral_connections = []
        
    for task_id, train_data in enumerate(train_loaders):
        
        criterion = nn.CrossEntropyLoss()
        train_loss, train_acc = [], []

        if task_id > 0:

            new_column = copy.deepcopy(model).to(device)
            
            task_lateral_connections = []
            for prev_col_idx in range(task_id):
                lateral = nn.Linear(feature_dim, feature_dim).to(device)
                task_lateral_connections.append(lateral)
            
            all_lateral_connections.append(task_lateral_connections)
            
            new_column.fc = nn.Linear(feature_dim * (task_id + 1), 2).to(device)
            columns.append(new_column)
        
        # Set all parameters to non-trainable
        for col_idx, column in enumerate(columns):
            for param in column.parameters():
                param.requires_grad = col_idx == task_id  # Only current column is trainable
        
        # Create optimizer for current column
        trainable_params = list(columns[task_id].parameters())
        
        # Add lateral connection parameters if they exist
        if task_id > 0:
            for lateral in all_lateral_connections[task_id-1]:
                trainable_params.extend(list(lateral.parameters()))
        
        optimizer = optim.Adam(trainable_params, lr=1e-3)
                
        for epoch in range(args.epochs):
            print(f"[INFO]: Epoch {epoch+1} of {args.epochs}")

            for i, col in enumerate(columns):
                if i == task_id:
                    col.train()
                else:
                    col.eval()

            train_epoch_loss, train_epoch_acc,  columns, all_lateral_connections = train(columns, all_lateral_connections, train_data, 
                                                optimizer, criterion, task_id, device)
            
            # valid_epoch_loss_p, valid_epoch_acc_p, valid_epoch_loss_n, valid_epoch_acc_n = validate(model, val_loaders[idx][0], val_loaders[idx][1],  
            #                                                 criterion, device)
            
            train_loss.append(train_epoch_loss)
            train_acc.append(train_epoch_acc)

            # valid_epoch_loss = (valid_epoch_loss_n + valid_epoch_loss_p)/2
            # valid_epoch_acc = (valid_epoch_acc_p + valid_epoch_acc_n)/2

            eval_dict = {f"train_epoch_loss_t{task_id}": train_epoch_loss, f"train_epoch_acc_t{task_id}": train_epoch_acc}
                        #  f"valid_epoch_loss_t{idx}": valid_epoch_loss, f"valid_epoch_acc_t{idx}": valid_epoch_acc}

            if wandb:
                wandb.log(eval_dict)

            rtpt.step()
        
        t_acc, t_acc_p, t_acc_n = test(columns, all_lateral_connections, test_loaders, -1, device)
        gt_acc, gt_acc_p, gt_acc_n = test(columns, all_lateral_connections, test_loaders_global, task_id, device)

        t_a_p = t_a_p + t_acc_p + gt_acc_p
        t_a_n = t_a_n + t_acc_n + gt_acc_n
        tt_acc = tt_acc + t_acc + gt_acc

    row = len(train_loaders)
    col_t = len(test_loaders) + len(test_loaders_global)

    
    t_acc_arr_p = np.asarray(t_a_p)
    test_acc_p = t_acc_arr_p.reshape(row, col_t)
    t_acc_arr_n = np.asarray(t_a_n)
    test_acc_n = t_acc_arr_n.reshape(row, col_t)
    t_acc_arr = np.asarray(tt_acc)
    test_acc = t_acc_arr.reshape(row, col_t)

    return  test_acc_p, test_acc_n, test_acc


def train(columns, all_lateral_connections, trainloader, optimizer, criterion, task_id, device=None):

    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0

    for data in tqdm(trainloader):
        counter += 1

        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        features = []
        
        # Get features from all previous columns
        for i in range(task_id):
            with torch.no_grad():
                feat = columns[i].extract_features(image)
                features.append(feat)
        
        # Get features from current column
        current_feat = columns[task_id].extract_features(image)
        features.append(current_feat)
        
        # Apply lateral connections and combine features
        if task_id > 0:
            lateral_feats = []
            for i in range(task_id):
                lateral_feat = all_lateral_connections[task_id-1][i](features[i])
                lateral_feats.append(lateral_feat)
            
            combined_feat = torch.cat([current_feat] + lateral_feats, dim=1)
        else:
            combined_feat = current_feat
        
        # Final classification
        outputs = columns[task_id].fc(combined_feat)
        # outputs = model(image)

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
    return loss, acc, columns, all_lateral_connections


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



def get_dataset(args):

    # the training transforms
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    train_t0 = args.train_path_task0
    train_t1 = args.train_path_task1
    train_t2 = args.train_path_task2

    val_t0 = args.val_path_task0
    val_t1 = args.val_path_task1
    val_t2 = args.val_path_task2

    test_t0 = args.test_path_task0
    test_t1 = args.test_path_task1
    test_t2 = args.test_path_task2

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
    
    traindataset0 = datasets.ImageFolder(root=train_t0, transform=transformer)
    valdataset0 = datasets.ImageFolder(root=val_t0, transform=transformer)
    positives = [i for i, (x, y) in enumerate(valdataset0) if  y == 1.0]
    negatives = [i for i, (x, y) in enumerate(valdataset0) if  y == 0.0]
    valdataset0_p = torch.utils.data.Subset(valdataset0, positives)
    valdataset0_n = torch.utils.data.Subset(valdataset0, negatives)

    testdataset0 = datasets.ImageFolder(root=test_t0, transform=transformer)
    positives = [i for i, (x, y) in enumerate(testdataset0) if  y == 1.0]
    negatives = [i for i, (x, y) in enumerate(testdataset0) if  y == 0.0]
    testdataset0_p = torch.utils.data.Subset(testdataset0, positives)
    testdataset0_n = torch.utils.data.Subset(testdataset0, negatives)

    traindataset1 = datasets.ImageFolder(root=train_t1, transform=transformer)
    valdataset1 = datasets.ImageFolder(root=val_t1, transform=transformer)
    positives = [i for i, (x, y) in enumerate(valdataset1) if  y == 1.0]
    negatives = [i for i, (x, y) in enumerate(valdataset1) if  y == 0.0]
    valdataset1_p = torch.utils.data.Subset(valdataset1, positives)
    valdataset1_n = torch.utils.data.Subset(valdataset1, negatives)

    testdataset1 = datasets.ImageFolder(root=test_t1, transform=transformer)
    positives = [i for i, (x, y) in enumerate(testdataset1) if  y == 1.0]
    negatives = [i for i, (x, y) in enumerate(testdataset1) if  y == 0.0]
    testdataset1_p = torch.utils.data.Subset(testdataset1, positives)
    testdataset1_n = torch.utils.data.Subset(testdataset1, negatives)

    traindataset2 = datasets.ImageFolder(root=train_t2, transform=transformer)
    valdataset2 = datasets.ImageFolder(root=val_t2, transform=transformer)
    positives = [i for i, (x, y) in enumerate(valdataset2) if  y == 1.0]
    negatives = [i for i, (x, y) in enumerate(valdataset2) if  y == 0.0]
    valdataset2_p = torch.utils.data.Subset(valdataset2, positives)
    valdataset2_n = torch.utils.data.Subset(valdataset2, negatives)

    testdataset2 = datasets.ImageFolder(root=test_t2, transform=transformer)
    positives = [i for i, (x, y) in enumerate(testdataset2) if  y == 1.0]
    negatives = [i for i, (x, y) in enumerate(testdataset2) if  y == 0.0]
    testdataset2_p = torch.utils.data.Subset(testdataset2, positives)
    testdataset2_n = torch.utils.data.Subset(testdataset2, negatives)

    train0_loader = DataLoader(dataset= traindataset0, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    train1_loader = DataLoader(dataset= traindataset1, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    train2_loader = DataLoader(dataset= traindataset2, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    val0_loader_p = DataLoader(dataset= valdataset0_p, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    val1_loader_p = DataLoader(dataset= valdataset1_p, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    val2_loader_p = DataLoader(dataset= valdataset2_p, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    val0_loader_n = DataLoader(dataset= valdataset0_n, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    val1_loader_n = DataLoader(dataset= valdataset1_n, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    val2_loader_n = DataLoader(dataset= valdataset2_n, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    test0_loader_p = DataLoader(dataset= testdataset0_p, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    test1_loader_p = DataLoader(dataset= testdataset1_p, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    test2_loader_p = DataLoader(dataset= testdataset2_p, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    test0_loader_n = DataLoader(dataset= testdataset0_n, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    test1_loader_n = DataLoader(dataset= testdataset1_n, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    test2_loader_n = DataLoader(dataset= testdataset2_n, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    train_loaders = [train0_loader, train1_loader, train2_loader]
    val_loaders = {0: [val0_loader_p, val0_loader_n], 1: [val1_loader_p, val1_loader_n], 2:[val2_loader_p, val2_loader_n]}
    test_loaders = {0: [test0_loader_p, test0_loader_n], 1: [test1_loader_p, test1_loader_n], 2:[test2_loader_p, test2_loader_n]}

    test_loaders_global = {0: [test_loader_p, test_loader_n]}

    return train_loaders, val_loaders, test_loaders, test_loaders_global



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Continual Coonfounding Dataset')

    parser.add_argument('--wandb', action='store_true', help='Log run to Weights and Biases.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rtpt', type=str, help='Use rtpt to set process name with given initials.', default='RK')
    parser.add_argument('--method', type=str, required=True, choices=METHODS)
    parser.add_argument('-results', '--results_dir', type=str, help='path to store results', required=True)
    parser.add_argument('-name', '--dataset_type', type=str, help='name of experiment')
    parser.add_argument('--save_model', action='store_true', help='save model weights', default=True)
    parser.add_argument('-e', '--epochs', type=int, default=50, help='number of epochs to train our network for')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='number of images per batch')
    parser.add_argument('-m', '--model_name', type=str, default='Resnet', help='select the model you would like to run training with')

    parser.add_argument('-p0', '--train_path_task0', type=str, help='path for train dataset of task 0')
    parser.add_argument('-p1', '--train_path_task1', type=str, help='path for train dataset of task 1')
    parser.add_argument('-p2', '--train_path_task2', type=str, help='path for train dataset of task 2')

    parser.add_argument('-pv0', '--val_path_task0', type=str, help='path for val dataset of task 0')
    parser.add_argument('-pv1', '--val_path_task1', type=str, help='path for val dataset of task 1')
    parser.add_argument('-pv2', '--val_path_task2', type=str, help='path for val dataset of task 2')

    parser.add_argument('-pt0', '--test_path_task0', type=str, help='path for test dataset of task 0')
    parser.add_argument('-pt1', '--test_path_task1', type=str, help='path for test dataset of task 1')
    parser.add_argument('-pt2', '--test_path_task2', type=str, help='path for test dataset of task 2')

    parser.add_argument('-gt0', '--test_path_global', type=str, help='path for global test')


    args = parser.parse_args()

    # seed = args.seed
    for seed in range(5):

        set_seed(seed)

        if args.rtpt is not None:
            from rtpt import RTPT
            rtpt = RTPT(name_initials=args.rtpt, experiment_name=f'{args.dataset_type}_{args.method}_{args.model_name}', max_iterations=args.epochs)
            rtpt.start()

        out_dir = os.path.dirname(args.results_dir)
        save_model = args.save_model
        dataset_type = f"{args.dataset_type}/{args.method}"
        model_name = args.model_name
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)
        print(args.method)


        if model_name == 'Resnet':
            model = ResNet(img_channels=3, num_layers=18, num_classes=2, block=BasicBlock).to(device)
            model_name = 'Resnet18'
            

        run_id =  wandb.util.generate_id()
        print(f'Sampled new wandb run_id {run_id}.')
        
        wandb.init(project='confounding_continual_dataset', name=os.path.dirname(args.dataset_type),
                    id=run_id, resume=True, config=args)
            
        
        print('Loading datasets...')
        train_loaders, val_loaders, test_loaders, test_loaders_global = get_dataset(args)
        
        
        test_acc_p, test_acc_n, test_acc = run(args.method, model, train_loaders, val_loaders, test_loaders, test_loaders_global, wandb, args, device)

        if save_model:  
            results = args.results_dir
            if not os.path.isdir(f'{results}/{dataset_type}/{model_name}/seed_{seed}'):
                os.makedirs(f'{results}/{dataset_type}/{model_name}/seed_{seed}')
            np.save(f'{results}/{dataset_type}/{model_name}/seed_{seed}/test_acc_p.npy', test_acc_p)
            np.save(f'{results}/{dataset_type}/{model_name}/seed_{seed}/test_acc_n.npy', test_acc_n)
            np.save(f'{results}/{dataset_type}/{model_name}/seed_{seed}/test_acc.npy', test_acc)
        