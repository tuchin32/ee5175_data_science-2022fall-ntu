import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
from tqdm import tqdm

from utils.options import args
import utils.common as utils

from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR as MultiStepLR
from torch.optim.lr_scheduler import StepLR as StepLR
from torch.utils.tensorboard import SummaryWriter

from data import dataPreparer

import warnings, math

warnings.filterwarnings("ignore")

device = torch.device(f"cuda:{args.gpus[0]}")
print(device)

checkpoint = utils.checkpoint(args)


def main():

    start_epoch = 0
    best_acc = 0.0
    
    # Create tensorboard
    tb_writer = SummaryWriter(args.job_dir)
 
    # Data loading
    print('=> Preparing data..')
 
    # data loader
    
    loader = dataPreparer.Data(args, 
                               data_path=args.src_data_path, 
                               label_path=args.src_label_path)
    
    data_loader = loader.loader_train
    data_loader_valid = loader.loader_valid
    data_loader_test = loader.loader_test
    print(len(data_loader_test))
    
    
    # Create model
    print('=> Building model...')

    # load training model
    model = import_module(f'model.{args.arch}').__dict__[args.model]().to(device)
    

    # Load pretrained weights
    if args.pretrained:
 
        ckpt = torch.load(os.path.join(checkpoint.ckpt_dir, args.source_file), map_location = device)
        state_dict = ckpt['state_dict']

        model.load_state_dict(state_dict)
        model = model.to(device)
        
    if args.inference_only:
        inference(args, data_loader_valid, model, args.output_file, save_csv=False, writer=tb_writer)
        inference(args, data_loader_test, model, args.output_file, save_csv=True)
        return

    param = [param for name, param in model.named_parameters()]
    
    # optimizer = optim.SGD(param, lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    optimizer = optim.Adam(param, lr = args.lr, weight_decay = args.weight_decay)
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma = args.lr_gamma)
    
    # Show the model architecture
    # print(model)

    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch: {epoch + 1}/{args.num_epochs}')
        
        scheduler.step(epoch)
        
        train(args, data_loader, model, optimizer, epoch, writer=tb_writer)
        
        valid_acc = valid(args, data_loader_valid, model)
   
        is_best = best_acc < valid_acc
        best_acc = max(best_acc, valid_acc)
        

        state = {
            'state_dict': model.state_dict(),
            
            'optimizer': optimizer.state_dict(),
            
            'scheduler': scheduler.state_dict(),
            
            'epoch': epoch + 1
        }
        checkpoint.save_model(state, epoch + 1, is_best)
        
    inference(args, data_loader_valid, model, args.output_file, save_csv=False, writer=tb_writer)
    inference(args, data_loader_test, model, args.output_file, save_csv=True)
    
    print(f'Best acc: {best_acc:.3f}\n')


  
       
def train(args, data_loader, model, optimizer, epoch, writer=None):
    losses = utils.AverageMeter()

    acc = utils.AverageMeter()

    criterion = nn.CrossEntropyLoss()
    
    num_iterations = len(data_loader)
    
    # switch to train mode
    model.train()
    pbar = tqdm(data_loader)
    for i, (inputs, targets, _) in enumerate(pbar, 1):
        
        num_iters = num_iterations * epoch + i

        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # train
        output = model(inputs)
        loss = criterion(output, targets)

        # optimize cnn
        loss.backward()
        optimizer.step()

        ## train weights        
        losses.update(loss.item(), inputs.size(0))
        
        ## evaluate
        prec1, _ = utils.accuracy(output, targets, topk = (1, 5))
        acc.update(prec1[0], inputs.size(0))

        
        if i % args.print_freq == 0:     
            print(
                'Epoch[{0}]({1}/{2}): \n'
                'Train_loss: {train_loss.val:.4f} ({train_loss.avg:.4f})\n'
                'Train acc {acc.val:.3f} ({acc.avg:.3f})\n'.format(
                epoch, i, num_iterations, 
                train_loss = losses,
                acc = acc))
            
    if writer:
        # visualizes loss and acc
        writer.add_scalar("Train Loss", losses.avg, epoch)
        writer.add_scalar("Train Accuracy", acc.avg, epoch)
                
      
 
def valid(args, loader_valid, model):
    losses = utils.AverageMeter()
    acc = utils.AverageMeter()

    criterion = nn.CrossEntropyLoss()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets, datafile) in enumerate(loader_valid, 1):
            
            inputs = inputs.to(device)
            targets = targets.to(device)
             
            preds = model(inputs)
            loss = criterion(preds, targets)
        
            # image classification results
            prec1, _ = utils.accuracy(preds, targets, topk = (1, 5))
            losses.update(loss.item(), inputs.size(0))
            acc.update(prec1[0], inputs.size(0))
 
    print(f'Validation acc {acc.avg:.3f}\n')

    return acc.avg
    

def inference(args, loader_test, model, output_file_name, save_csv=False, writer=None):
    outputs = []
    datafiles = []
    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets, datafile) in enumerate(loader_test, 1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
          
            preds = model(inputs)
    
            _, output = preds.topk(1, 1, True, True)
            
            outputs.extend(list(output.reshape(-1).cpu().detach().numpy()))
            
            datafiles.extend(list(datafile))
            
    
    if save_csv == True:
        output_file = dict()
        output_file['image_name'] = datafiles
        output_file['label'] = outputs
        
        output_file = pd.DataFrame.from_dict(output_file)
        output_file.to_csv(output_file_name, index = False)
    
    if writer:
        gt_df = pd.read_csv('../digit/valid.csv')
        conf_mat = confusion_matrix(outputs, gt_df['label'])
        
        fig = plt.figure()
        plt.title('Confusion matrix')
        sn.set(font_scale=1) # for label size
        sn.heatmap(conf_mat, annot=True, annot_kws={"size": 5}, fmt='d', cmap='bone') # font size  
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.savefig(f'{args.job_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')     
        plt.show()
        
        writer.add_figure('confusion_matrix', fig)

  

if __name__ == '__main__':
    main()

