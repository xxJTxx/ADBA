import argparse
import copy
import os
import shutil
import sys
import warnings
import torchvision.models as models
import numpy as np
import math
import pdb
import torch
import wandb
import logging
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from helpers.datasets import partition_data
from helpers.utils import get_dataset, average_weights, DatasetSplit, KLDiv, setup_seed, test, kldiv, add_trigger, test_trigger_accuracy
from models.generator import Generator
from models.nets import CNNCifar, CNNMnist, CNNCifar100
from models.resnet import resnet18
from models.vit import deit_tiny_patch16_224
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from loaders import get_cifar10_loaders, get_cifar100_loaders, get_svhn_loaders, get_mnist_loaders,get_cifar10_loaders_sub
from similarity_check import activated_neuron_similarity

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10, help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=1, help="number of users: K")
    parser.add_argument('--num_poison_users', type=int, default=1, help="number of poison users: K")
    parser.add_argument('--frac', type=float, default=1, help='the fraction of clients: C')
    #parser.add_argument('--local_ep', type=int, default=100, help="the number of local epochs: E")
    #parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.5)')
    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--iid', type=int, default=1, help='Default set to IID. Set to 0 for non-IID.')


    # Data Free
    parser.add_argument('--adv', default=0, type=float, help='scaling factor for adv loss')
    parser.add_argument('--bn', default=0, type=float, help='scaling factor for BN regularization')
    parser.add_argument('--oh', default=0, type=float, help='scaling factor for one hot loss (cross entropy)')
    parser.add_argument('--act', default=0, type=float, help='scaling factor for activation loss used in DAFL')
    parser.add_argument('--save_dir', default='run/synthesis', type=str)
    parser.add_argument('--partition', default='dirichlet', type=str)
    parser.add_argument('--beta_partition', default=0.5, type=float, help=' If beta is set to a smaller value, then the partition is more unbalanced')
    
    
    # BackDoor
    parser.add_argument('--beta_backdoor', default=0.3, type=float, help=' If beta is set to a smaller value, then the partition is more unbalanced')
    parser.add_argument('--alpha_backdoor', default=1, type=float, help=' If beta is set to a smaller value, then the partition is more unbalanced')
    parser.add_argument('--target_label_backdoor',default=0,type=int,help='target label for poison ')

    # Basic
    parser.add_argument('--lr_g', default=1e-3, type=float, help='initial learning rate for generation')
    parser.add_argument('--T', default=1, type=float)
    parser.add_argument('--g_steps', default=20, type=int, metavar='N', help='number of iterations for generation')
    parser.add_argument('--batch_size', default=256, type=int, metavar='N', help='number of total iterations in each epoch')
    parser.add_argument('--nz', default=256, type=int, metavar='N', help='number of total iterations in each epoch')
    parser.add_argument('--synthesis_batch_size', default=256, type=int)
    # Misc
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
    parser.add_argument('--type', default="pretrain", type=str, help='seed for initializing training.')
    parser.add_argument('--model', default="res", type=str, help='seed for initializing training.')
    parser.add_argument('--other', default="", type=str, help='seed for initializing training.')
    parser.add_argument('--txtpath', default="", type=str, help='txt for some ducument')
    parser.add_argument('--miu', default=0.1, type=float, help='scaling factor for normalization')

    parser.add_argument('--msg', default="", type=str, help='Addtional infor for directory')
    args = parser.parse_args()
    return args

def get_model(args):
    if args.model == "mnist_cnn":
        global_model = CNNMnist().cuda()
    elif args.model == "fmnist_cnn":
        global_model = CNNMnist().cuda()
    elif args.model == "cnn":
        global_model = CNNCifar().cuda()
    elif args.model == "svhn_cnn":
        global_model = CNNCifar().cuda()
    elif args.model == "cifar100_cnn":
        global_model = CNNCifar100().cuda()
    elif args.model == "res":
        # global_model = resnet18()
        global_model = resnet18(num_classes=10)

    elif args.model == "vit":
        global_model = deit_tiny_patch16_224(num_classes=1000,
                                             drop_rate=0.,
                                             drop_path_rate=0.1)
        global_model.head = torch.nn.Linear(global_model.head.in_features, 10)
        global_model = global_model.cuda()
        global_model = torch.nn.DataParallel(global_model)
    return global_model

# Custom Loss function that take into two models water_model and train_model, and return the mean squared error between output of certain layers
def relu_neu_loss(water_relu, train_relu):
        w_denominator=water_relu.detach()
        t_denominator=train_relu.detach()       
        w_n = torch.where(water_relu !=0, water_relu/w_denominator, water_relu)
        t_n = torch.where(train_relu !=0, train_relu/t_denominator, train_relu)
        return torch.mean(torch.pow((w_n - t_n), 2))

# Knowledge Distillation Loss    
def loss_fn_kd(outputs, labels, teacher_outputs, alpha, temperature):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = DEVICE
    T = temperature
    labels = torch.tensor(labels, dtype=torch.long)
    KD_loss = nn.KLDivLoss().cuda(device)(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels).cuda(device) * (1. - alpha)

    return KD_loss

# Custom Loss function
def custom_loss1(water_model, train_model, trigger=1e-2):
        # Denominator take abs value to maintain the sign
        w_denominator=torch.abs(water_model.detach())
        t_denominator=torch.abs(train_model.detach()) 
        
        # Creating a new tensor where negative/positive values are changed to -1/1, wheras 0 becomes 0+trigger
        water_one = torch.FloatTensor(water_model.size()).type_as(water_model)
        mask_w = water_model!=0
        water_one[mask_w] = water_model[mask_w]/w_denominator[mask_w]
        mask_w = water_model==0
        water_one[mask_w] = water_model[mask_w] + trigger

        # Creating a new tensor where negative/positive values are changed to -1/1, wheras 0 remains 0
        mask = train_model==0
        train_one = torch.FloatTensor(train_model.size()).type_as(train_model)
        train_one[mask] = train_model[mask]
        mask = train_model!=0
        train_one[mask] = train_model[mask]/t_denominator[mask]
        
        # To give the right direction of gradient when the value from fix tensor is 0
        water_one[mask_w] = (torch.sign(train_one[mask_w])-1)*water_one[mask_w]

       
        ''' for idx in range(len(train_one)):
          print(f"w_o:{water_one[idx]}~~~t_o:{train_one[idx].item(),}") '''
        return torch.mean(torch.pow((1 + water_one*train_one), 2))



def draw_curve(loss1, loss2, loss3, acc, poison, main_ratio, output_dir):
    if len(loss1) == len(loss2) == len(loss3) == len(acc) == len(poison) == len(main_ratio):
        
        # Create the figure and axis
        fig, ax1 = plt.subplots(figsize=(10,4))
        
        # Plot lists 1, 2, and 3 on the left y-axis
        ax1.plot(loss1, label='New Loss')
        ax1.plot(loss2, label='Main Loss')
        ax1.plot(loss3, label='Relu Loss')
        ax1.set_ylabel('Loss')
        ax1.tick_params(axis='y')   

        # Create a twin y-axis for lists 4, 5, and 6
        ax2 = ax1.twinx()
        ax2.plot(acc, label='Acc', color='r')
        ax2.plot(poison, label='Poison', color='y')
        ax2.plot(main_ratio, label='Main r', color='k')
        new_ratio = [1 - x for x in main_ratio]
        ax2.plot(new_ratio, label='New r', color='c')
        ax2.set_ylabel('acc.')
        ax2.tick_params(axis='y', colors='r')

        # Set labels and title
        ax1.set_xlabel('Epoch')
        ax1.set_title('Performance')
        
        # Add legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, bbox_to_anchor = (1.05, 0), loc = 3, borderaxespad = 0)

        # Adjust the spacing between the y-axes
        fig.tight_layout()
                
        plt.savefig(os.path.join(output_dir, 'Performance.png'))
        #plt.show()
    else:
        print('Input lists should have same length, please check again!')
        print(f"loss1: {len(loss1)}")
        print(f"loss2: {len(loss2)}")
        print(f"loss3: {len(loss3)}")
        print(f"acc: {len(acc)}")
        print(f"poisson: {len(poison)}")
        print(f"main_r: {len(main_ratio)}")

# Define the training loop
def start_train_kd_loss1(dataset, subset_rate, train_model, water_model, optimizer, device, response, mask, trigger, num_epochs=50, new_loss_r=0, default_loss_r=1, main_loss='CE', layer_output=None, layer_input=None, hook_layer=None, alpha = 0.9, temprature = 20, output_dir=None):
    
    for epoch in range(num_epochs):
        
        if epoch == 0:
            
            # Generate subset data loader based on dataset
            if dataset == 'cifar10':
                train_loader, val_loader, test_loader = get_cifar10_loaders_sub(subset_rate)
            elif dataset == 'cifar100':
                train_loader, val_loader, test_loader = get_cifar100_loaders()
            elif dataset == 'svhn':
                train_loader, val_loader, test_loader = get_svhn_loaders()
            
            # Record accuaracy after every epoch
            water_test_acc = []
            train_test_acc = []
            water_query_acc = []
            train_query_acc = []
            neuron_loss_after_epoch = [] 
            task_loss_after_epoch = [] 
            # TO CHECK RELU RESULT
            relu_neuron_loss_after_epoch =[]
            acc = []
            main_ratio = []
            poison = []
            
            if layer_output is not None:
                # Record the input of every hooked layer
                water_relu = []
                train_relu = []            
                # Define the hook function
                w_hooks = [] # list of hook handles, to be removed when you are done
                t_hooks = []
                hook_flag = False
                def water_hook(module, input, output):
                    if hook_flag:
                        nonlocal water_relu
                        water_relu.append(output)   
                def train_hook(module, input, output):
                    if hook_flag:
                        nonlocal train_relu
                        train_relu.append(output)

                # Hook the function onto conv1 and conv2 of layer1~layer4 of both models.
                for layer,_ in water_model.named_children():
                    if layer in hook_layer:
                        s = getattr(water_model, layer)
                        for idx in range(len(s)):
                            for name,_ in s[idx].named_children():
                                if name in layer_output:
                                    w_hooks.append(getattr(s[idx], name).register_forward_hook(water_hook))
                
                for layer,_ in train_model.named_children():
                    if layer in hook_layer:
                        s = getattr(train_model, layer)
                        for idx in range(len(s)):
                            for name,_ in s[idx].named_children():
                                if name in layer_output:
                                    t_hooks.append(getattr(s[idx], name).register_forward_hook(train_hook))
                
                print(f"{len(w_hooks)} and {len(t_hooks)} layers of output are being recorded on water/train model.")

            if layer_input is not None:    
                # TO CHECK RELU RESULT
                water_relu1 = []
                train_relu1 = []        
                # TO CHECK RELU RESULT
                w_hooks1 = [] # list of hook handles, to be removed when you are done
                t_hooks1 = []
                def water_hook1(module, input, output):
                    if hook_flag:
                        nonlocal water_relu1
                        water_relu1.append(input)   
                def train_hook1(module, input, output):
                    if hook_flag:
                        nonlocal train_relu1
                        train_relu1.append(input)
                # Hook the function onto conv1 and conv2 of layer1~layer4 of both models.
                for layer,_ in water_model.named_children():
                    if layer in hook_layer:
                        s = getattr(water_model, layer)
                        for idx in range(len(s)):
                            for name,_ in s[idx].named_children():
                                if name in layer_input:
                                    w_hooks1.append(getattr(s[idx], name).register_forward_hook(water_hook1))
                
                for layer,_ in train_model.named_children():
                    if layer in hook_layer:
                        s = getattr(train_model, layer)
                        for idx in range(len(s)):
                            for name,_ in s[idx].named_children():
                                if name in layer_input:
                                    t_hooks1.append(getattr(s[idx], name).register_forward_hook(train_hook1))
                
                print(f"{len(w_hooks1)} and {len(t_hooks1)} layers of input are being recorded on water/train model.")
        
                    
        if epoch == 0 : 
            print("Target label: ", response)   
            print("Train model main/query acc eval...")
            main_acc, _ = test(train_model, test_loader)
            trigger_acc = round(test_trigger_accuracy(test_loader=test_loader, model= train_model, target_label=response, mask=mask, trigger=trigger),2)
            #query_acc = round(model_on_queryset(train_model, query, response, device).item(),2)
            train_test_acc.append([epoch,main_acc,trigger_acc])
            acc.append(main_acc)
            poison.append(trigger_acc)
            if callable(default_loss_r):
                main_ratio.append(default_loss_r(epoch))
            else:
                main_ratio.append(1)      
        
         
        print(f"===============================Now in epoch {epoch+1}...===============================")

        # To track performance of the two losses
        ave_neu_loss_per_epoch = 0.0
        ave_task_loss_per_epoch = 0.0
        
        # TO CHECK RELU RESULT
        ave_relu_neu_loss_per_epoch = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            
            hook_flag = True
            # Reset the lists
            water_relu = []
            train_relu = []
            water_relu1 = []
            train_relu1 = []
            
            optimizer.zero_grad()
            images = batch[0]
            labels = batch[1].long()         
            images, labels = images.to(device), labels.to(device)
            
            train_model.train()
            # If not set to eval(), the model will still changing even if not being opt.step(), due to the change of BN layer during the forward process
            water_model.eval() 
            
            outputs = train_model(images)
            with torch.no_grad():
                outputs_water = water_model(images) 
            
            # For checking hook functions work correctly
            if layer_input and layer_output:
                if not water_relu and not train_relu:
                    print("No value stored in water_relu and train_relu...")
                    breakpoint()
                elif len(water_relu) != len(train_relu):
                    print("The length of water_relu and train_relu are not equal...")
                    breakpoint()
                if not water_relu1 and not train_relu1:
                    print("No value stored in water_relu1 and train_relu1...")
                    breakpoint()
                elif len(water_relu1) != len(train_relu1):
                    print("The length of water_relu1 and train_relu1 are not equal...")
                    breakpoint()    
            
             # Reset new_loss
            new_loss = 0.0
            # Sum up the loss of conv1 and conv2 of layer1~layer4 of both models
            for idx in range(len(water_relu)):
                new_loss += custom_loss1(water_relu[idx][0].detach(), train_relu[idx][0]) / len(water_relu)

            """ if batch_idx % 5 == 0:
                print(f"{batch_idx+1} batch neuron loss: {new_loss.item()}")     """
            
            # TO CHECK RELU RESULT
            relu_new_loss = 0.0
            # Sum up the loss of conv1 and conv2 of layer1~layer4 of both models
            for idx in range(len(water_relu1)):
                relu_new_loss += relu_neu_loss(water_relu1[idx][0].detach(), train_relu1[idx][0].detach()) / len(water_relu1)
 
            """ if batch_idx % 5 == 0:
                print(f"{batch_idx+1} batch relu neuron loss: {relu_new_loss.item()}")   """
            
            if main_loss == 'KD':
                kd_loss = loss_fn_kd(outputs, labels, outputs_water, alpha=alpha, temperature=temprature)
            elif main_loss == 'KDhard':
                epoch_with_batch = epoch + (batch_idx+1) / len(train_loader)
                lr_scheduler = lambda t: np.interp([t],\
                    [0, 100, 100, 150, 150, 200],\
                    [0.1, 0.1, 0.01, 0.01, 0.001, 0.001])[0]
                if lr_scheduler is not None:
                    lr_new = lr_scheduler(epoch_with_batch)
                    for param_group in opt.param_groups:
                        param_group.update(lr=lr_new)    
                outputs_water_hard = outputs_water.topk(1, dim=1).indices.squeeze()
                kd_loss = F.cross_entropy(outputs, outputs_water_hard)
            else:
                kd_loss = F.cross_entropy(outputs, labels)
            
             
            """ if batch_idx % 5 == 0:
                print(f"{batch_idx+1} batch kd loss: {kd_loss.item()}") """      
            
            # Combine both losses
            # When default_loss_r is passed in with fixed value
            if not callable(default_loss_r):
                loss = (default_loss_r)*kd_loss + (new_loss_r)*(new_loss)
                """ if batch_idx % 5 == 0:
                    print(f"{batch_idx+1} batch {default_loss_r}:{new_loss_r} combined loss: {loss.item()}") """
            # When default_loss_r is passed in with non_fixed ratio (new_loss_r will be replaced with the number calculated below and ignore what was passing into this function)
            else:
                #loss_mul = 100/(epoch+1)+4 if epoch > 20 else 10
                loss_mul = 10
                loss = loss_mul*(10*(default_loss_r(epoch))*kd_loss + (1-default_loss_r(epoch))*(new_loss))
                """ if batch_idx % 5 == 0:
                    print(f"{batch_idx+1} batch non_fixed {default_loss_r(epoch)}:{1-default_loss_r(epoch)} combined loss: {loss.item()}") """
            
            ave_neu_loss_per_epoch += new_loss.item()
            ave_task_loss_per_epoch += kd_loss.item()
            # TO CHECK RELU RESULT
            ave_relu_neu_loss_per_epoch += relu_new_loss.item()
            
            loss.backward()
            optimizer.step()

        if callable(default_loss_r):
            print(f"Non-fixed loss ratio: {default_loss_r(epoch)}:{1-default_loss_r(epoch)}")
        else:
            print(f"Fixed ratio: {default_loss_r}:{new_loss_r}")
        
        if epoch % 1 == 0:    
            ave_task_loss_per_epoch /= len(train_loader)
            ave_neu_loss_per_epoch /= len(train_loader)
            neuron_loss_after_epoch.append(round(ave_neu_loss_per_epoch,4))
            task_loss_after_epoch.append(round(ave_task_loss_per_epoch,4))
            # TO CHECK RELU RESULT
            ave_relu_neu_loss_per_epoch /= len(train_loader)
            relu_neuron_loss_after_epoch.append(round(ave_relu_neu_loss_per_epoch,4))
            # Duplicate the loss of the first epoch to represent the loss before training
            if epoch == 0:
                neuron_loss_after_epoch.append(round(ave_neu_loss_per_epoch,4))
                task_loss_after_epoch.append(round(ave_task_loss_per_epoch,4))
                relu_neuron_loss_after_epoch.append(round(ave_relu_neu_loss_per_epoch,4))
        
        
        # Validate the model
        print("Validation Process...")
        train_model.eval()
        hook_flag = False # Turn off the hook for validation
        test(train_model, val_loader)
        
        
        if epoch % 1 == 0:   
            #Testing train model
            print("Train model main/query acc eval...")
            main_acc, _ = test(train_model, test_loader)
            trigger_acc = round(test_trigger_accuracy(test_loader=test_loader, model= train_model, target_label=response, mask=mask, trigger=trigger),2)
            train_test_acc.append([epoch+1,main_acc,trigger_acc])
            acc.append(main_acc)
            if callable(default_loss_r):
                main_ratio.append(default_loss_r(epoch))
            else:
                main_ratio.append(1)       
            poison.append(trigger_acc)
        
        if epoch == 0 or epoch == num_epochs-1:    
            print("Water model main/query acc eval...")
            main_acc, _ = test(water_model, test_loader)
            trigger_acc = round(test_trigger_accuracy(test_loader=test_loader, model= water_model, target_label=response, mask=mask, trigger=trigger),2)
            water_test_acc.append(f"{main_acc}/{trigger_acc}")
        
    # Remove hooks for model after used and reset the handle lists
    for handle in w_hooks:
        handle.remove()
    for handle in t_hooks:
        handle.remove()
    w_hooks=[]
    t_hooks=[]
    
    # TO CHECK RELU RESULT
    for handle in w_hooks1:
        handle.remove()
    for handle in t_hooks1:
        handle.remove()
    w_hooks1=[]
    t_hooks1=[]
    
    draw_curve(neuron_loss_after_epoch,task_loss_after_epoch,relu_neuron_loss_after_epoch, acc, poison, main_ratio, output_dir)
    
    print('===============================Finished Training===============================')
    print('===============================Finished Training===============================')
    print('===============================Finished Training===============================')
    """ print(f"Neuron loss after every epoch: {neuron_loss_after_epoch}")
    print(f"K.D. loss after every epoch: {task_loss_after_epoch}")
    # TO CHECK RELU RESULT
    print(f"Relu Neuron loss after every epoch: {relu_neuron_loss_after_epoch}") """    
    
    a = int(len(neuron_loss_after_epoch)/21)
    a1 = len(neuron_loss_after_epoch)%21
        
    logging.info(f"Neuron loss after every epoch:")
    for idx in range(a):
        logging.info(neuron_loss_after_epoch[idx*21:idx*21+21])
    logging.info(neuron_loss_after_epoch[a*21:])
    logging.info(f"K.D. loss after every epoch:")
    for idx in range(a):
        logging.info(task_loss_after_epoch[idx*21:idx*21+21])
    logging.info(task_loss_after_epoch[a*21:])
    # TO CHECK RELU RESULT
    logging.info(f"Relu Neuron loss after every epoch:")
    for idx in range(a):
        logging.info(relu_neuron_loss_after_epoch[idx*21:idx*21+21])
    logging.info(relu_neuron_loss_after_epoch[a*21:])
    
    return train_test_acc, train_query_acc, water_test_acc, water_query_acc       



if __name__ == '__main__':

    #torch.cuda.empty_cache()
    #force_cudnn_initialization()

    args = args_parser()
    
    ########################### Hyperparameters setting ###########################
    dataset = 'cifar10'
    subset_rate = 0.5 # 0~1
    epoch = 199
    main_loss_ratio = 10 # >=0
    new_loss_ratio = 0# >=0
    device = DEVICE
    print("Device: ", device)
    hook_layer = ['layer3','layer4'] #'layer1','layer2',
    layer_input = ['conv2'] # List of layers that input will be used as relu loss input
    layer_output = ['conv1'] # List of layers that output will be used as custom loss input
    # Non-fixed loss ratio
    scheduler1=[0  ,   5,   5,    8,    8,    40]
    scheduler2=[0.8, 0.9, 0.9, 0.95, 0.95,  0.95]
    main_loss_scheduler = lambda t: np.interp([t],\
            scheduler1,\
            scheduler2)[0]
    ratio_type = 'fix' #'fix' or 'scheduler'
    main_loss_type = 'CE' # 'CE', 'KD', 'KDhard'
    alpha_kl = 1 # 0~1, Ratio for KL-divergence, (1-alpha) for C.E. in K.D.
    temperature = 20 # >=1, temp for KL-divergence
    opt_type = 'SGD' # 'Adam' or 'SGD'
    opt_lr = 0.001
    opt_mo = 0.9
    using_checkpoint = 0 #0:false, 1:true
    target_label = 4 #args.target_label_backdoor
    ###############################################################################    
    
    # Create directory
    exp_name = "_".join([dataset, ratio_type, main_loss_type, opt_type, args.msg])
    output_dir = os.path.join('hyperpara', exp_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Configure the logging module
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S', 
        level=logging.INFO,
        handlers=[
                logging.FileHandler(os.path.join(output_dir, "log.log")),
                logging.StreamHandler()
                ])
    
    # Set loss ratio type
    if ratio_type == 'fix':
        the_main_task_r = main_loss_ratio
    elif ratio_type == 'scheduler':
        the_main_task_r = main_loss_scheduler
    else:
        print("Choose the predifined type of ratio.")
        breakpoint()
    
    
    mask_origin = torch.load('./saved/cifar10_res_200_0.4_0.3_4_/pretrain/mask.pt')
    trigger_origin = torch.load('./saved/cifar10_res_200_0.4_0.3_4_/pretrain/pattern.pt')
    local_weights = torch.load('./saved/cifar10_res_200_0.4_0.3_4_/pretrain/checkpoint.pkl')
    
    # Create a copy of mask and trigger, just to make sure mask/trigger remain unchange
    mask = mask_origin.detach()
    trigger = trigger_origin.detach()
    
    # Create model in load checkpoint if needed
    water_model = get_model(args)
    water_model.load_state_dict(local_weights[0])
    
    # dropout-based initialization
    layerwise_ratio = [0.01, 0.01, 0.03, 0.09, 0.27, 0.10]
    train_model = get_model(args)
    new_state = train_model.cpu().state_dict()
    old_state = copy.deepcopy(local_weights[0])
    # old_state.pop('fc.weight')
    # old_state.pop('fc.bias')
    num_layers = 0
    for key in local_weights[0].keys():
        if key.find('bn') != -1 or key.find('shortcut.1') != -1:
            continue
        if key.endswith('.weight') or key.endswith('.bias'):
            p = layerwise_ratio[0]
            if key.startswith('layer1'):
                p = layerwise_ratio[1]
                print("Layer 1 dropout initialized.")
            elif key.startswith('layer2'):
                p = layerwise_ratio[2]
                print("Layer 2 dropout initialized.")
            elif key.startswith('layer3'):
                p = layerwise_ratio[3]
                print("Layer 3 dropout initialized.")
            elif key.startswith('layer4'):
                p = layerwise_ratio[4]
                print("Layer 4 dropout initialized.")
            elif key.startswith('fc'):
                p = layerwise_ratio[5]
                print("FC dropout initialized.")

            # if key.startswith('fc'):
            #     p = 1
            # elif key.find('shortcut') != -1:
            #     p = 1
            #     # p = 1 - (num_layers - 3) * 0.01
            #     print(key, p)
            # else:
            #     p = num_layers * 0.01
            #     print(key, p)
            #     num_layers += 1
            mask_one = torch.ones(old_state[key].shape) * (1 - p)
            mask = torch.bernoulli(mask_one)
            # masked_weight = old_state[key] * mask * (1/(1-p)) + new_state[key] * (1 - mask)
            masked_weight = old_state[key] * mask + new_state[key] * (1 - mask)     # 1 copy, 0 random
            old_state[key] = masked_weight
    train_model.load_state_dict(old_state, strict=False)   
    
    logging.info("Using CVPR23 BCU initialized on Train model")
    
    """ if using_checkpoint:
        #print("Start with checkpoint")
        logging.info("Start with checkpoint")
        train_model.load_state_dict(local_weights[0])
    else:
        #print("Start with initialed weight")
        logging.info("Start with initialed weight") """
    
    
    optimizer = torch.optim.SGD(train_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    """ # Load the optimizer from checkpoint
    if opt_type == 'SGD':
        opt = torch.optim.SGD(train_model.parameters(), lr=opt_lr, momentum=opt_mo)
    elif opt_type == 'Adam':
        opt = torch.optim.Adam(train_model.parameters())
    else:
        print("Choose the predifined type of optimizer.")
        breakpoint() """
    
    train_model.to(device)
    water_model.to(device)
    mask.to(device)
    trigger.to(device)
    
    
        
    # Start training
    train_test_acc, train_query_acc, water_test_acc, water_query_acc = start_train_kd_loss1(dataset, subset_rate, train_model, water_model, opt, device, target_label, mask, trigger, epoch, new_loss_ratio, the_main_task_r, main_loss_type , layer_output, layer_input, hook_layer, alpha_kl, temperature, output_dir)
    
    # Print the results
    """ if ratio_type == "fix":    
        print(f"===============================Training on {subset_rate} of {dataset} with {main_loss_type}/new loss ratio {main_loss_ratio}/{new_loss_ratio} and opt {opt_type} for {epoch} epochs.===============================")
    elif ratio_type == "scheduler":
        print(f"===============================Training on {subset_rate} of {dataset} with non-fixed ratio on {main_loss_type} and opt {opt_type} for {epoch} epochs.===============================")
    print("Train model Test Acc:", train_test_acc)
    print(f"Water model Test/Query Acc:{water_test_acc}")    
    
    print(f"########################### Hyperparameters setting ###########################")
    print(f"hook_layer = {hook_layer}, layer_input = {layer_input}, layer_out = {layer_output}")
    if ratio_type == "scheduler":
        print(f"main_loss_scheduler_t = {scheduler1}")
        print(f"main_loss_scheduler_r = {scheduler2}")
    if main_loss_type == 'KD':
        print(f"Alpha/Temprature: {alpha_kl}/{temperature}")
    print(f"###############################################################################") """
    
    b = int(len(train_test_acc)/9)
    b1 = len(train_test_acc)%9
      
    if ratio_type == "fix":
        logging.info(f"===============================Training on {subset_rate} of {dataset} with {main_loss_type}/new loss ratio {main_loss_ratio}/{new_loss_ratio} and opt {opt_type} for {epoch} epochs.===============================")    
    elif ratio_type == "scheduler":
        logging.info(f"===============================Training on {subset_rate} of {dataset} with non-fixed ratio on {main_loss_type} and opt {opt_type} for {epoch} epochs.===============================")
    logging.info(f"Train model Test Acc:")
    for idx in range(b):
        logging.info(train_test_acc[idx*9:idx*9+9])
    logging.info(train_test_acc[b*9:])
    logging.info(f"Water model Test/Query Acc:{water_test_acc}")
    
    logging.info(f"########################### Hyperparameters setting ###########################")
    logging.info(f"hook_layer = {hook_layer}, layer_input = {layer_input}, layer_out = {layer_output}")
    if ratio_type == "scheduler":
        logging.info(f"main_loss_scheduler_t = {scheduler1}")
        logging.info(f"main_loss_scheduler_r = {scheduler2}")
    if opt_type == 'SGD':
        logging.info(f"SGD setting lr/momen: {opt_lr}/{opt_mo}")
    if main_loss_type == 'KD':
        logging.info(f"KD setting Alpha/Temprature: {alpha_kl}/{temperature}")
    logging.info(f"###############################################################################")
    
    l1s, l2s, l3s, l4s, ts = activated_neuron_similarity(dataset, water_model, device, mask, trigger)
    logging.info("Neuron Similarity of Watered model:")
    logging.info(f"L1 similarity: {l1s}/ L2 similarity: {l2s}/ L3 similarity: {l3s}/ L4 similarity: {l4s}/ Total similarity: {ts}")
    
    l1s, l2s, l3s, l4s, ts = activated_neuron_similarity(dataset, train_model, device, mask, trigger)
    logging.info("Neuron Similarity of Trained model:")
    logging.info(f"L1 similarity: {l1s}/ L2 similarity: {l2s}/ L3 similarity: {l3s}/ L4 similarity: {l4s}/ Total similarity: {ts}")
    
    