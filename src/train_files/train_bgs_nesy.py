import os
import argparse
import random
import math

import numpy as np

import torch
torch.set_num_threads(6)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import wandb
from model_nesy import *
from utils import *
from buffer import Buffer, BufferGDumbBalanced
from torch.nn import functional as F

METHODS = ['bgs']


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


def get_dataloader(train_dataset, get_balanced_loader=False,
                       batch_size=None, drop_last=True):

    kwargs = {'num_workers': 0, 'pin_memory': True}
    batch_size = batch_size if batch_size is not None else args.batch_size
    sampler = None
    shuffle = True
    if get_balanced_loader:
        from torch.utils.data.sampler import WeightedRandomSampler
        weights = train_dataset.make_weights()
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=drop_last,
                                                shuffle=shuffle, sampler=sampler, **kwargs)
    
    return train_loader

def _unfix_model(model):
    model.eval()
    for name, param in model.named_parameters():
        param.requires_grad = True

def _fix_model(model):
    model.eval()
    for name, param in model.named_parameters():
        if not name.startswith('set_cls.dec.3'):
            param.requires_grad = False
        print(name, param.requires_grad)


def fill_buffer(buffer, train_dataset, t):

    train_loader = get_dataloader(train_dataset, drop_last=False) 
    for samples in train_loader:
        data, target= samples

        batch_size = data.shape[0]
        task_label = torch.full((batch_size,), t, dtype=torch.int)
        task_label = task_label.cuda()

        groups = target.detach().clone()
        groups[:] = (2 * t) + groups[:] 

        stop_flag = buffer.add_data(examples=data, labels=target,
                                            task_labels=task_label, groups=groups)
        if stop_flag:
            break


def train_step(model, trainloader, buff_loader, optimizer, criterion, device):
    model.img2state_net.eval()
    model.set_cls.train()
    torch.set_grad_enabled(True)
    buff_iter = None
    if buff_loader is not None:
        buff_iter = iter(buff_loader)
    counter = 0
    train_running_loss = 0
    train_running_correct = 0
    for samples in trainloader:
        counter += 1
        if len(samples) == 2:
            data, target = samples
        else:
            data, target, _, __ = samples

        data, target = data.cuda(), target.cuda()

        if buff_iter is not None:
            try:
                buf_data, buf_target, _, _ = next(buff_iter)
            except StopIteration:
                buff_iter = iter(buff_loader)
                buf_data, buf_target, _, _ = next(buff_iter)

            buf_data, buf_target = buf_data.cuda(), buf_target.cuda()
            
            data = torch.cat((data, buf_data))
            target = torch.cat((target, buf_target))

        output, _ = model(data)
        loss_ce = criterion(output, target)

        train_running_loss += loss_ce.item()
        _, preds = torch.max(output.data, 1)
        train_running_correct += (preds == target).sum().item()

        optimizer.zero_grad()
        loss_ce.backward(retain_graph=True)
        optimizer.step()


    # loss and accuracy for the complete epoch
    loss = train_running_loss / counter
    acc = 100. * (train_running_correct / len(trainloader.dataset))
    return loss, acc


def train_step_bgs(model, trainloader, optimizer, criterion, device):
    model.img2state_net.eval()
    model.set_cls.train()
    torch.set_grad_enabled(True)

    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    loss = 0

    for _, samples in enumerate(trainloader):
        counter += 1

        if len(samples) == 2:
            x, y = samples
        else:
            x, y, _, __ = samples

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        logits, _ = model(x)
        loss = criterion(logits, y)

        train_running_loss += loss.item()
        _, preds = torch.max(logits.data, 1)
        train_running_correct += (preds == y).sum().item()
        
        loss.backward(retain_graph=True)
        optimizer.step()


    # loss and accuracy for the complete epoch
    loss = train_running_loss / counter
    acc = 100. * (train_running_correct / len(trainloader.dataset))
    return loss, acc


def run(model, train_loaders, val_loaders, test_loaders, test_loaders_global, wandb, args, device):

    t_a_p, t_a_n, tt_acc = [], [], []
    
    buffer = BufferGDumbBalanced(args.buffer_size, device, 3)
    buffer.add_new_classes(2)

    for idx, train_data in enumerate(train_loaders):
        
        _unfix_model(model)
        optimizer = torch.optim.Adam(
        [p for name, p in model.named_parameters() if p.requires_grad and 'set_cls' in name], lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.000001)

        # lists to keep track of losses and accuracies
        train_loss, train_acc = [], []

        for epoch in range(args.epochs):
            print(f"[INFO]: Epoch {epoch+1} of {args.epochs}")

            buf_loader = None if idx == 0 else get_dataloader(buffer, drop_last=False, batch_size=args.batch_size)
            train_epoch_loss, train_epoch_acc = train_step(model, train_data, buf_loader, optimizer, criterion, device)
            
            train_loss.append(train_epoch_loss)
            train_acc.append(train_epoch_acc)

            scheduler.step()

        fill_buffer(buffer, train_data.dataset, idx)
        _fix_model(model)
        optimizer = torch.optim.Adam(
        [p for name, p in model.named_parameters() if p.requires_grad and 'set_cls' in name], lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.000001)
        buf_loader = get_dataloader(buffer, drop_last=False, batch_size=args.batch_size) 

        for epoch in range(args.epochs):
            train_epoch_loss, train_epoch_acc = train_step_bgs(model, buf_loader, optimizer, criterion, device)
            scheduler.step()

        # load best model for final evaluation
        model_test = NeSyConceptLearner(n_classes=args.n_imgclasses, n_slots=args.n_slots, n_iters=args.n_iters_slot_att,
                                n_attr=args.n_attr, n_set_heads=args.n_heads, set_transf_hidden=args.set_transf_hidden,
                                category_ids=args.category_ids, device=args.device)
        
        model_test = model_test.to(args.device)
        model_test.load_state_dict(model.state_dict())
        print("\nModel loaded from checkpoint for final evaluation\n")

        t_acc, t_acc_p, t_acc_n = test(model_test, test_loaders, device)
        gt_acc, gt_acc_p, gt_acc_n = test(model_test, test_loaders_global, device)

        t_a_p = t_a_p + t_acc_p + gt_acc_p
        t_a_n = t_a_n + t_acc_n + gt_acc_n
        tt_acc = tt_acc + t_acc + gt_acc

        rtpt.step()

    row = len(train_loaders)
    col_t = len(test_loaders) + len(test_loaders_global)

    t_acc_arr_p = np.asarray(t_a_p)
    test_acc_p = t_acc_arr_p.reshape(row, col_t)
    t_acc_arr_n = np.asarray(t_a_n)
    test_acc_n = t_acc_arr_n.reshape(row, col_t)
    t_acc_arr = np.asarray(tt_acc)
    test_acc = t_acc_arr.reshape(row, col_t)

    return  test_acc_p, test_acc_n, test_acc



def validate(model, testloader_p, testloader_n, criterion, device):
    model.eval()
    torch.set_grad_enabled(False)
    
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
            outputs, _ = model(image)

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
            outputs, _ = model(image)

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
    torch.set_grad_enabled(False)
    
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

                output, _ = model(image)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct_p += pred.eq(labels.view_as(pred)).sum().item()

            for i, data in enumerate(testloader_n):
                
                image, labels = data
                image = image.to(device)
                labels = labels.to(device)

                output, _ = model(image)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct_n += pred.eq(labels.view_as(pred)).sum().item()

        acc_p = 100. * correct_p / len(testloader_p.dataset)
        acc_n = 100. * correct_n / len(testloader_n.dataset)
        
        M_a_p.append(acc_p)
        M_a_n.append(acc_n)
        M_a.append((acc_n+acc_p)/2)

    return M_a, M_a_p, M_a_n



def get_dataset(args):

    transformer = transforms.Compose(
            [transforms.Resize((128, 128)),
             CustomTransform()
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
    parser.add_argument('--val_flag', action='store_true', default=False)
    parser.add_argument('--save_model', action='store_true', help='save model weights', default=True)
    parser.add_argument('-e', '--epochs', type=int, default=50, help='number of epochs to train our network for')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='number of images per batch')
    parser.add_argument('-m', '--model_name', type=str, default='Resnet', help='select the model you would like to run training with')
    parser.add_argument('-swa', '--slot_attention_weights', type=str, help='pre-trained weights for the slot attention module', required=True)
    parser.add_argument('-m_size', '--buffer_size', type=int, default=100, help='memory buffer size')

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

    parser.add_argument(
        "--lr", type=float, default=1e-2, help="Outer learning rate of model"
    )
    parser.add_argument(
        "--l2_grads", type=float, default=1, help="Right for right reason weight"
    )

    # Slot attention params
    parser.add_argument('--n-slots', default=10, type=int,
                        help='number of slots for slot attention module')
    parser.add_argument('--n-iters-slot-att', default=3, type=int,
                        help='number of iterations in slot attention module')
    parser.add_argument('--n-attr', default=18, type=int,
                        help='number of attributes per object')


    args = parser.parse_args()

    seed = args.seed
    set_seed(seed)

    if args.rtpt is not None:
        from rtpt import RTPT
        rtpt = RTPT(name_initials=args.rtpt, experiment_name=f'{args.dataset_type}_{args.method}', max_iterations=args.epochs)
        rtpt.start()

    out_dir = os.path.dirname(args.results_dir)
    save_model = args.save_model
    dataset_type = f"{args.dataset_type}/{args.method}"
    model_name = args.model_name
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(args.method)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.device=device

    args.n_imgclasses = 2
    args.n_iters_slot_att=3
    args.n_attr=18
    args.n_heads=4
    args.set_transf_hidden=128
    args.class_weights = torch.ones(args.n_imgclasses)/args.n_imgclasses
    args.classes = np.arange(args.n_imgclasses)
    args.category_ids = np.array([3, 6, 8, 10, 18])

    model = NeSyConceptLearner(n_classes=2, n_slots=10, n_iters=3, n_attr=18, n_set_heads=4, set_transf_hidden=128,
                                category_ids = [3, 6, 8, 10, 17], device='cuda').to(device)
    model_name = 'Slot_Attention'

    log = torch.load(args.slot_attention_weights, map_location=torch.device(args.device))
    model.img2state_net.load_state_dict(log['weights'], strict=True)
    print("Pretrained slot attention model loaded!")

    model = model.to(args.device)

    model_name = 'Slot_Attention'

    run_id =  wandb.util.generate_id()
    print(f'Sampled new wandb run_id {run_id}.')
    
    wandb.init(project='confounding_continual_dataset', name=os.path.dirname(args.dataset_type),
                id=run_id, resume=True, config=args)
    
    print('Loading datasets...')
    train_loaders, val_loaders, test_loaders, test_loaders_global = get_dataset(args)

    results = args.results_dir
    if not os.path.isdir(f'{results}/{dataset_type}/{model_name}/seed_{seed}'):
        os.makedirs(f'{results}/{dataset_type}/{model_name}/seed_{seed}')
    
    path=f'{results}/{dataset_type}/{model_name}/seed_{seed}'
    test_acc_p, test_acc_n, test_acc = run(model, train_loaders, val_loaders, test_loaders, test_loaders_global, wandb, args, device)
    
    np.save(f'{results}/{dataset_type}/{model_name}/seed_{seed}/test_acc_p.npy', test_acc_p)
    np.save(f'{results}/{dataset_type}/{model_name}/seed_{seed}/test_acc_n.npy', test_acc_n)
    np.save(f'{results}/{dataset_type}/{model_name}/seed_{seed}/test_acc.npy', test_acc)
    