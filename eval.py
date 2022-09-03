import torch
import torch.nn as nn
import torchvision
def double_conv01(in_channels, out_channels,f_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    )

def double_conv11(in_channels, out_channels,f_size,p_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, f_size,padding=p_size,stride=2),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        
        nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    ) 

def double_conv_u1(in_channels, out_channels,f_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, f_size, padding='same'),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    ) 


def trans_1(in_channels, out_channels,f_size,st_size):
    return nn.Sequential(
       nn.ConvTranspose2d(in_channels,out_channels, kernel_size=f_size, stride=st_size),
       nn.BatchNorm2d(num_features=out_channels),
       nn.ReLU(inplace=True),
    ) 


class m_unet_33_1(nn.Module):

    def __init__(self, input_channels=3):
        super().__init__()
                
        self.dconv_down1 = double_conv01(input_channels, 64,(3,3))
        self.dconv_down2 = double_conv11(64, 128,(3,3),(1,1))
        self.dconv_down3 = double_conv11(128, 256,(3,3),(1,1))
        self.dconv_down4 = double_conv01(256, 512,(3,3))
        self.dconv_down5 = double_conv01(512, 512,(3,3))
        
        #self.up0 = trans_1(512,256, 2,2)
        self.up1 = trans_1(256,256,  2,2)
        self.up2 = trans_1(128, 128, 2,2)
        #self.up3 = trans_1(128, 64, 2,2)
        
        
        self.m = nn.Dropout(p=0.10)

        
        self.dconv_up0 = double_conv_u1(512, 512,(3,3))
        self.dconv_up1 = double_conv_u1(512 + 512, 512,(3,3))
        self.dconv_up2 = double_conv_u1(512+256, 256,(3,3))
        self.dconv_up3 = double_conv_u1(256+128,128,(3,3))
        
        self.dconv_up4 = double_conv_u1(192,64,(3,3))
        
        self.conv_last = nn.Conv2d(64, 1, 1)
        self.activation = torch.nn.Sigmoid()
        
        
    def forward(self, x_in):
        
        conv1 = self.dconv_down1(x_in)      
        conv2 = self.dconv_down2(conv1)
        conv3 = self.dconv_down3(conv2)
        conv4 = self.dconv_down4(conv3)
        conv5 = self.dconv_down5(conv4)

        # # ## decoder ####
        
        conv5=self.m(conv5)
        u0=self.dconv_up0(conv5)
        u0 = torch.cat([u0, conv4], dim=1) 

        u1=self.dconv_up1(u0)
        
        u1 = torch.cat([u1, conv3], dim=1) 
        
        u2=self.dconv_up2(u1)
        u2=self.up1(u2)
        u2 = torch.cat([u2, conv2], dim=1) 
        
        
        u3=self.dconv_up3(u2)
        u3=self.up2(u3)
        u3 = torch.cat([u3, conv1], dim=1) 
        u3=self.dconv_up4(u3)
        
        out=self.conv_last(u3)
        return self.activation(out)
    
import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader

NUM_WORKERS=0
PIN_MEMORY=True

class Dataset_(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        
        
        image = np.load(img_path,allow_pickle=True, fix_imports=True)
        
        gen_img=np.zeros([3,image.shape[0],image.shape[1]]) 
        
        mean=np.mean(image,keepdims=True)
        std=np.std(image,keepdims=True)
        image=(image-mean)/std
        
        img_o=image.copy()    ## orignal img
        
        image[np.where(image<0)]=0   #### no negatives
        
        hitss=np.histogram(image, bins=5)
        
        value_=hitss[1][1]
        img1=np.zeros([image.shape[0],image.shape[1]])  
        img1[np.where(image>=value_)]=1  
        new_img=img1*img_o                 #### from hists
        
        gen_img[0,:,:]=image
        gen_img[1,:,:]=img_o
        gen_img[2,:,:]=new_img
        mask = np.load(mask_path,allow_pickle=True, fix_imports=True)
        mask[np.where(mask>0)]=1
        
        image=np.expand_dims(image, axis=0)
        mask=np.expand_dims(mask, axis=0)
            
        return gen_img,mask,self.images[index]
    
def Data_Loader( test_dir,test_maskdir,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    
    test_ids = Dataset_( image_dir=test_dir, mask_dir=test_maskdir)

    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    
    return data_loader

import torch.nn as nn
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm


batch_size=64

val_imgs='/rds/user/akr54/hpc-work/patches1/val/img/'
val_masks='/rds/user/akr54/hpc-work/patches1/val/gt/'


val_loader=Data_Loader(val_imgs,val_masks,batch_size)


model=m_unet_33_1()

LEARNING_RATE=0.0001
optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999),lr=LEARNING_RATE)

weights_paths="/home/akr54/patches/train4_8Reduced.pth.tar"

from sklearn.metrics import confusion_matrix
def Evaluation_Metrics(pre,gt):
    pre=pre.flatten() 
    gt=gt.flatten()  
    tn, fp, fn, tp=confusion_matrix(gt,pre,labels=[0,1]).ravel()
    
    iou=tp/(tp+fn+fp) 
    dice=2*tp/(2*tp + fp + fn)
    return iou,dice,tp,tn,fp,fn 
    
def check_accuracy(loader, model, device=DEVICE):
    dice_score1=0
    dice_score2=0
    
    TP=0
    TN=0
    FP=0
    FN=0
    
    loop = tqdm(loader)
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, t1,label) in enumerate(loop):
            data = data.to(device=DEVICE,dtype=torch.float)
            t1 = t1.to(device=DEVICE,dtype=torch.float)

            p1=model(data)
            p1 = (p1 > 0.5).float()
            
            
            
            dice_score1 += (2 * (t1 * p1).sum()) / (
                (t1 + p1).sum() + 1e-8
            )
            
            p1=p1.cpu()
            t1=t1.cpu()
            
            iou,dice,tp,tn,fp,fn=Evaluation_Metrics(p1,t1)
            
            TP=TP+tp
            FP=FP+fp
            TN=TN+tn
            FN=FN+fn
            
            dice_score2=dice_score2+dice
            
    print(f"Dice score for Segmentation of scars1: {dice_score1/len(val_loader)}")
    print(f"Dice score for Segmentation of scars2: {dice_score2/len(val_loader)}")
    print(f"TP   : {TP}")
    print(f"TN   : {TN}")
    print(f"FP   : {FP}")
    print(f"FN   : {FN}")
    

def eval_():
    model.to(device=DEVICE,dtype=torch.float)
    checkpoint = torch.load(weights_paths,map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    check_accuracy(val_loader, model, device=DEVICE)
    
if __name__ == "__main__":
    eval_()
    
    
    
    
