"""
Train implementation for HP - FCN.

Code Implementation: Stavros Papadopoulos
May 2021
"""

from numpy.core.fromnumeric import argmax
from numpy.core.shape_base import block
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim import optimizer 
from torch.utils.data import Dataset
import numpy as np
import cv2
import os 
from torchvision import transforms
import torchvision.io.image
from Datasets.custom_dataset import custom_dataset
from torch.utils.data import Dataset, DataLoader, random_split
import resnet
from log.visdom_logger import visdom_logger
from log.plot import plot_img, plot_loss_val, plot_map 
from tqdm import tqdm
import matplotlib.pyplot as plt
#from sklearn.metrics import precision_recall_fscore_support
from ignite.metrics import Recall, Precision, IoU, ConfusionMatrix, confusion_matrix
import HRNet.lib.models.seg_hrnet as seg_hrnet
import HRNet.lib.config as my_config
from iou import get_IoU as IoU 
from torchvision.utils import save_image
from torchsummary import summary

import matplotlib.pyplot as plt
from HRNet.lib.models import seg_hrnet

import argparse 

g_arg_parser = argparse.ArgumentParser()
g_arg_parser.add_argument("--cfg",type=str,default=R"C:\Users\stavr\Desktop\thesis\Src\HRNet\experiments\cityscapes\seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml") 
g_arg_parser.add_argument('opts',type=str,default=None,nargs=argparse.REMAINDER)
g_my_args = g_arg_parser.parse_args()


class InpaintingForensics():
    def __init__(self):
        
        self.vlogger = visdom_logger('inpainting forensics')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(torch.cuda.is_available())
        # create the cnn network
        if net_type=='hp':
            self.net = resnet.ResNet(block=resnet.block)
        else:
            self.seg_hrnet_config = my_config.config
            my_config.update_config(self.seg_hrnet_config,args=g_my_args)
            self.net = seg_hrnet.HighResolutionNet(self.seg_hrnet_config)
        self.net.to(self.device)
        self.batch_size=4
        #set train, val and test dataset
        train_dataset = custom_dataset(data_src='did')
        train_dataset, val_dataset = random_split(train_dataset, (round(0.9*len(train_dataset)), round(0.1*len(train_dataset))))
        test_dataset= custom_dataset(data_src='default')


        #loads data from  batch size = 1 when we use defactp as train data_src 
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

    def train(self):
        '''
        Steps of Training
        1)Make a forward pass through the network
            #logits = model(images)
        2)Use the network output to calculate the loss. 
            # Calculate the loss with the logits and the labels
            # loss = criterion(logits, labels)
        3)Perform a backward pass through the network with loss.backward() to calculate the gradients
        4)Take a step with the optimizer to update the weights
        '''
        self.n_epochs = 10
        self.optimizer=optim.Adam(self.net.parameters(),lr=0.0001, weight_decay=0.00001) 
        self.criterion = nn.BCELoss()
        train_losses, vld_losses = [], []
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5)


        #best_auc = 0
        plot_counter_train=0
        plot_counter_vld=0

        for epoch in range(self.n_epochs):
            tot_train_loss = 0
            print(f"Epoch: {epoch}. Learning rate {scheduler.get_lr()}\r")

            for i, data  in tqdm(enumerate(self.train_loader, 0)):
                #print(f"iteration {i}\r")
                # pass image and ground truth mask
                image, gt_mask = data[0].to(self.device), data[1].to(self.device)
    
                # zero the optimizer gradients
                self.optimizer.zero_grad()

                #forward pass & loss calculation
                output = self.net(image)
                #the following 3 only for hrnet
                if net_type=='hrnet':
                    output = torch.sigmoid(output)
                    h_init,w_init= image.shape[2], image.shape[3]
                    output = F.interpolate(output, size=(h_init,w_init), mode='nearest')

                # calculate loss
                loss = self.criterion(output, gt_mask)
                tot_train_loss += loss.item()

                # bacward propagation 
                loss.backward()

                # optimize step
                self.optimizer.step()

                #print (running_loss)
                if  i % 100 == 0: #printe every 500 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, tot_train_loss / (i+1)))
                    plot_img(self.vlogger, image, 'input_image', mode='Train')
                    plot_map(self.vlogger,output,'predicted mask',mode='Train')  
                    plot_map(self.vlogger,gt_mask,'GT mask',mode='Train')
                    plot_loss_val(self.vlogger, loss.item(), plot_counter_train, ' Train Loss' ,mode= 'Train')
                plot_counter_train += 1
                     
            else:
                #plot_counter_vld = 0
                tot_vld_loss = 0
                with torch.no_grad():
                    for i, data  in tqdm(enumerate(self.val_loader, 0)):
                        image, gt_mask = data[0].to(self.device), data[1].to(self.device)
                        vld_output = self.net(image)
                        vld_output = torch.sigmoid(vld_output)
                        h_init,w_init= image.shape[2], image.shape[3]
                        vld_output = F.interpolate(vld_output, size=(h_init,w_init), mode='nearest')
                        loss = self.criterion(vld_output, gt_mask)
                        tot_vld_loss += loss.item()

                        if i % 100 == 0: #print every 500 mini-batches
                            print('[%d, %5d] loss: %.3f' %
                            (epoch + 1, i + 1, tot_vld_loss / (i+1)))
                            plot_img(self.vlogger, image, 'VLD input_image', mode='Evaluation')
                            plot_map(self.vlogger,vld_output,'VLD predicted mask', mode='Evaluation')  
                            plot_map(self.vlogger,gt_mask,'VLD GT mask', mode='Evaluation')
                            plot_loss_val(self.vlogger, loss.item(), plot_counter_vld, ' VLD Loss ', mode='Evaluation')   
                        plot_counter_vld += 1


                # Get mean loss to enable comparison between train and test sets
                train_loss = tot_train_loss / len(self.train_loader)
                vld_loss = tot_vld_loss / len(self.val_loader)

                # At completion of epoch
                train_losses.append(train_loss)
                vld_losses.append(vld_loss)
                #plot_loss_val(self.vlogger, train_losses[-1], epoch + 1, ' Training Loss ', mode='Train')   
                #plot_loss_val(self.vlogger, vld_losses[-1], epoch + 1, ' VLD Loss ', mode='Evaluation')   


                print("Epoch: {}/{}.. ".format(epoch + 1 , self.n_epochs),
                    "Training Loss: {:.3f}.. ".format(train_losses[-1]),
                    "vld Loss: {:.3f}.. ".format(vld_losses[-1]))
                scheduler.step()
            #if epoch> 30:
             #scheduler.step()

        print ('Finished Training\r')
        print (f" Training Loss {train_loss}\r")
        print (f" Valid Loss {vld_loss}\r")

        
        #path = R'C:\Users\stavr\Desktop\thesis\Src\savedmodels\hrnet_trained_defacto_e50.ckp'
        #torch.save(self.net.state_dict(),path)

    def test(self):
        path = './savedmodels/hrnet_trained_did_e50.ckp'
        #gt_path_to_save='C:/Users/stavr/Desktop/thesis/Experiments/hp/DID/50 epoch/test predictions/th_50/gt_masks/'
        pr_path_to_save='C:/Users/stavr/Desktop/thesis/Experiments/hp/DID/50 epoch/test predictions/th_75/pr_masks/'
        if net_type=='hp':
            self.net = resnet.ResNet(block=resnet.block).cuda()
        else:
            self.seg_hrnet_config = my_config.config
            my_config.update_config(self.seg_hrnet_config,args=g_my_args)
            self.net = seg_hrnet.HighResolutionNet(self.seg_hrnet_config)
        print(self.net)
        self.net.to(self.device)
        self.net.load_state_dict(torch.load(path))
        self.net.eval()
        summary(self.net,(1,1,224,224))
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            recall = Recall()
            precision = Precision()
            recall.reset()
            precision.reset()
            total_iou=0
            for i,data in tqdm(enumerate(self.test_loader, 0)):
                print(i, end='\r')
                images, gt_mask = data[0].to(self.device), data[1].to(self.device)
                # calculate outputs by running images through the network
                outputs = self.net(images)
                outputs = torch.sigmoid(outputs)
                if net_type=='hrnet':
                    h_init,w_init= images.shape[2], images.shape[3]
                    outputs = F.interpolate(outputs, size=(h_init,w_init), mode='nearest')
                outputs = torch.round(abs(outputs-0.75+0.5))
                gt_mask = torch.round(abs(gt_mask-0.75+0.5))
                #outputs[outputs>=0.65] = 1
                #outputs[outputs<0.65] = 0
                plot_map(self.vlogger,outputs,'Output mask', mode='Evaluation') 
                plot_map(self.vlogger,gt_mask,'GT mask', mode='Evaluation') 
                recall.update((outputs, gt_mask))
                precision.update((outputs, gt_mask))
                iou_instict=IoU(outputs,gt_mask)
                total_iou= iou_instict +total_iou

                if i % 100 == 0: #print every 500 mini-batches
                    plot_img(self.vlogger, images, 'Test input_image', mode='Evaluation')
                    plot_map(self.vlogger,outputs,'Test predicted mask', mode='Evaluation')  
                    plot_map(self.vlogger,gt_mask,'Test GT mask', mode='Evaluation')
                    #plot_loss_val(self.vlogger, recall, i, ' Recall ', mode='Evaluation')
                    
                    pr_save_path = os.path.join(pr_path_to_save,f'{i}'+'_pred.png')
                    #gt_save_path = os.path.join(gt_path_to_save,f'{i}'+'_gtmask.png')
                    outputs=outputs[0]
                    save_image(outputs,pr_save_path)
                    #gt_mask=gt_mask[0]
                    #save_image(gt_mask,gt_save_path)
                    
            total_recall = recall.compute()
            total_precision = precision.compute()
            F1 = (total_precision * total_recall * 2 / (total_precision + total_recall)).mean()
            total_iou= total_iou/2000
            print(f'total recall {total_recall}, precision {total_precision}, F1 {F1}, IoU {total_iou}')

if __name__=='__main__':
    mode='train'
    net_type='hrnet'
    exp=InpaintingForensics()
    if mode == 'train':
        exp.train()
    elif mode =='test':
        exp.test()
    




