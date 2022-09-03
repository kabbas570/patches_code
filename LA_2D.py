import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import cv2
import albumentations as A

transform2 = A.Compose([
    A.Resize(width=320, height=320)
])

NUM_WORKERS=0
PIN_MEMORY=True

class Dataset_(Dataset):
    def __init__(self, image_dir, mask_dir,transform2=transform2):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transform2 = transform2

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        
        image = np.load(img_path,allow_pickle=True, fix_imports=True)
        sp_dim=image.shape[1]
        
        mean=np.mean(image,keepdims=True)
        std=np.std(image,keepdims=True)
        image=(image-mean)/std

        mask = np.load(mask_path,allow_pickle=True, fix_imports=True)
        mask[np.where(mask>0)]=1
        
        
        
        if sp_dim==576:
            temp=np.zeros([640,640])
            temp[32:608, 32:608] = image
            image=temp
            
            
            temp1=np.zeros([640,640])    
            temp1[32:608, 32:608] = mask
            mask=temp1
            
        if self.transform2 is not None:
            
            augmentations2 = self.transform2(image=image)

            image2 = augmentations2["image"]
            
            image2=np.expand_dims(image2,0)
            image=np.expand_dims(image,0)
            mask=np.expand_dims(mask,0)

        return image,image2,mask,self.images[index],sp_dim
    
def Data_Loader( test_dir,test_maskdir,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    
    test_ids = Dataset_( image_dir=test_dir, mask_dir=test_maskdir)

    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    
    return data_loader



batch_size=1


val_imgs='/home/akr54/from_data/test_imgs/'
val_masks=r'/home/akr54/from_data/test_LA/'

# val_imgs=r'C:\Users\Abbas Khan\Desktop\data_analysis\valid\img'
# val_masks=r'C:\Users\Abbas Khan\Desktop\data_analysis\valid\gt'

test_loader=Data_Loader(val_imgs,val_masks,batch_size)
print(len(test_loader))

# a=iter(test_loader)
# a1=next(a)
# img1=a1[0][0,0,:,:].numpy()
# img2=a1[1][0,0,:,:].numpy()
# gt2=a1[2][0,0,:,:].numpy()

### models 
import torch
import torch.nn as nn

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
class Model_LA(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(Model_LA, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // 2)
        self.up1 = Up(1024, 512 // 2)
        self.up2 = Up(512, 256 // 2)
        self.up3 = Up(256, 128 // 2)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)
        self.activation = torch.nn.Sigmoid()
        
        self.up_ = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1,x1_):
        ##encoder 1 ##
        x1 = self.inc(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        ## encoder 222# ##a
        x1_ = self.inc(x1_)
        x2_ = self.down1(x1_)
        x3_ = self.down2(x2_)
        x4_ = self.down3(x3_)
        x5_ = self.down4(x4_)
        

        ##decoder###
        x5_ = self.up_(x5_)
        x55=x5+x5_ 
        x4_ = self.up_(x4_)
        x44=x4+x4_
        x3_ = self.up_(x3_)
        x33=x3+x3_
        x2_ = self.up_(x2_)
        x22=x2+x2_
        x1_ = self.up_(x1_)
        x11=x1+x1_
         
        x = self.up1(x55, x44)
        x = self.up2(x, x33)
        x = self.up3(x, x22)
        x = self.up4(x, x11)
        
        x = self.outc(x)
        return self.activation(x) 


 ###Loading Wights #####
 
import torch.nn as nn
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm



model=Model_LA()
LEARNING_RATE=0.0001
optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999),lr=LEARNING_RATE)



weights_paths="/home/akr54/patches/m_unet4_Final.pth.tar"
path_0='/home/akr54/from_data/LA_pre/0/'
path_1='/home/akr54/from_data/LA_pre/1/'
path_2='/home/akr54/from_data/LA_pre/2/'
path_3='/home/akr54/from_data/LA_pre/3/'


def check_accuracy(loader, model, device=DEVICE):
    loop = tqdm(loader)
    model.eval()
    with torch.no_grad():
        for batch_idx, (img1,img2,t1,label,sp_dim) in enumerate(loop):
            
            img1 = img1.to(device=DEVICE,dtype=torch.float)
            img2 = img2.to(device=DEVICE,dtype=torch.float)
            t1 = t1.to(device=DEVICE,dtype=torch.float)
            
            filename0 = os.path.join(path_0,label[0])
            filename1 = os.path.join(path_1,label[0])
            filename2 = os.path.join(path_2,label[0])
            filename3 = os.path.join(path_3,label[0])
            
            ##### 00 #####
            p0=model(img1,img2)
            p0=p0.cpu()
            p0=p0[0,0,:,:]
            if sp_dim==576:
                temp=np.zeros([640,640])
                temp= p0[32:608, 32:608] 
                p0=temp
            np.save(filename0, p0)
            
            ##### 90 #####
            img1=torch.rot90(img1, 1, [2,3])
            img2=torch.rot90(img2, 1, [2,3])
            
            p1=model(img1,img2)
            p1=p1.cpu()
            p1=p1[0,0,:,:]
            if sp_dim==576:
                temp=np.zeros([640,640])
                temp= p1[32:608, 32:608] 
                p1=temp
            np.save(filename1, p1)
            
            ##### 180 #####
            img1=torch.rot90(img1, 2, [2,3])
            img2=torch.rot90(img2, 2, [2,3])
            
            p2=model(img1,img2)
            p2=p2.cpu()
            p2=p2[0,0,:,:]
            if sp_dim==576:
                temp=np.zeros([640,640])
                temp= p2[32:608, 32:608] 
                p2=temp
            np.save(filename2, p2)
            
            ##### 270 #####
            img1=torch.rot90(img1, 3, [2,3])
            img2=torch.rot90(img2, 3, [2,3])
            
            p3=model(img1,img2)
            p3=p3.cpu()
            p3=p3[0,0,:,:]
            if sp_dim==576:
                temp=np.zeros([640,640])
                temp= p3[32:608, 32:608] 
                p3=temp
            np.save(filename3, p3)
               

def eval_():
    model.to(device=DEVICE,dtype=torch.float)
    checkpoint = torch.load(weights_paths,map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    check_accuracy(test_loader, model, device=DEVICE)
    
if __name__ == "__main__":
    eval_()