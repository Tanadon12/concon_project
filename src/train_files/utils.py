import os 
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
matplotlib.style.use('ggplot')
import seaborn as sns
sns.set(font_scale=2.0)




def save_plots(TL, VL, TA, VA, model_name, concat_type):
    """
    Function to save the loss and accuracy plots to disk.
    """

    if concat_type == 3:
        c = 'All three at once'
        t = 3

        train_acc0 = TA[0]
        train_loss0 = TL[0]

        val_acc0 = VA[0]
        val_loss0 = VL[0]

        fig, axes = plt.subplots(2, 1, figsize=(10,7))
        fig.suptitle(c)
        axes[0].plot(TA[0], color='green', linestyle='-', label='train accuracy')
        axes[0].plot(VA[0], color='blue', linestyle='-', label='validation accuracy')
        axes[0].set_ylabel('Accuracy')


        axes[1].plot(TL[0], color='orange', linestyle='-', label='train loss')
        axes[1].plot(VL[0], color='red', linestyle='-', label='validation loss')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss')

        fig.legend()
        

    elif concat_type < 3:

        if concat_type == 2:
            t = 2
            c = 'Additional class every training step'
        elif concat_type == 1:
            t = 1
            c = 'Training on only a single class at a time'

        train_acc0 = TA[0]
        train_acc1 = TA[1]
        train_acc2 = TA[2]
        train_loss0 = TL[0]
        train_loss1 = TL[1]
        train_loss2 = TL[2]

        val_acc0 = VA[0]
        val_acc1 = VA[1]
        val_acc2 = VA[2]
        val_loss0 = VL[0]
        val_loss1 = VL[1]
        val_loss2 = VL[2]
    
        fig, axes = plt.subplots(2, 3, figsize=(10,7))
        axes[0][0].plot(TA[0], color='green', linestyle='-', label='train accuracy')
        axes[0][0].plot(VA[0], color='blue', linestyle='-', label='validation accuracy')
        axes[0][0].set_ylabel('Accuracy')

        axes[0][1].plot(TA[1], color='green', linestyle='-')
        axes[0][1].plot(VA[1], color='blue', linestyle='-')

        axes[0][2].plot(TA[2], color='green', linestyle='-') 
        axes[0][2].plot(VA[2], color='blue', linestyle='-')

        axes[1][0].plot(TL[0], color='orange', linestyle='-', label='train loss')
        axes[1][0].plot(VL[0], color='red', linestyle='-', label='validation loss')
        axes[1][0].set_xlabel('Epochs')
        axes[1][0].set_ylabel('Loss')

        axes[1][1].plot(TL[1], color='orange', linestyle='-')
        axes[1][1].plot(VL[1], color='red', linestyle='-')
        axes[1][1].set_xlabel('Epochs')

        axes[1][2].plot(TL[2], color='orange', linestyle='-')
        axes[1][2].plot(VL[2], color='red', linestyle='-')
        axes[1][2].set_xlabel('Epochs')

        fig.suptitle(c, fontsize=16)
        fig.legend()
        

    else: 
        print('give legitimate concat type')


def plot_heatmaps(acc, loss, model_name):

    xticks = ['Task 0', 'Task 1', 'Task 2']
    yticks = ['Task 0', 'Task 1', 'Task 2']
    fig, (ax1, ax2) = plt.subplots(1,2)
    a = sns.heatmap(acc, cmap='crest', annot=True, fmt='.2f', xticklabels=xticks, yticklabels=yticks, ax=ax1)
    a.set(xlabel ="Train", ylabel = "Validate", title ='Accuracy')
    b = sns.heatmap(loss, cmap='crest', annot=True, fmt='.2f', xticklabels=xticks, yticklabels=yticks, ax=ax2)
    b.set(xlabel ="Train", title ='Loss')
    plt.tight_layout()
    
    fig.clf()


def plot_accuracy_confusion_matrix(acc, model_name, dataset_type, file_name, xticks, yticks, xlabel='Validation', ylabel='Train', vmin=40, vmax=100):

    plt.subplots(figsize=(15, 10))
    heat_acc = sns.heatmap(acc, cmap='Blues_r', annot=True, fmt='.2f', xticklabels=xticks, yticklabels=yticks, vmin=vmin, vmax=vmax)
    heat_acc.set(xlabel=xlabel, ylabel=ylabel)
    figure_acc = heat_acc.get_figure()    
   

def plot_loss_confusion_matrix(loss, model_name, dataset_type, file_name,  xticks, yticks, xlabel='Validation', ylabel='Train', vmin=0, vmax=1):
    
    plt.subplots(figsize=(15, 10))
    heat_loss = sns.heatmap(loss, cmap='crest', annot=True, fmt='.2f', xticklabels=xticks, yticklabels=yticks, vmin=vmin, vmax=vmax)
    heat_loss.set(xlabel=xlabel, ylabel=ylabel)
    figure_loss = heat_loss.get_figure()    
    