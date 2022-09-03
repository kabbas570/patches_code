import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import torchvision

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
        new_img=img1*image                 #### from hists
        
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
from Early_Stopping import EarlyStopping
import matplotlib.pyplot as plt
from tqdm import tqdm


batch_size=128

train_imgs='/rds/user/akr54/hpc-work/patches1/train/img/'
train_masks='/rds/user/akr54/hpc-work/patches1/train/gt/'


val_imgs='/rds/user/akr54/hpc-work/patches1/val/img/'
val_masks='/rds/user/akr54/hpc-work/patches1/val/gt/'


train_loader=Data_Loader(train_imgs,train_masks,batch_size)
val_loader=Data_Loader(val_imgs,val_masks,batch_size)


print(len(train_loader))
print(len(val_loader))


avg_train_losses = []   # losses of all training epochs
avg_valid_losses = []  #losses of all training epochs
avg_valid_DS = []  # all training epochs


NUM_EPOCHS=200
LEARNING_RATE=0.0001

from models_1 import m_unet_33_1

#def train_fn(loader_train,loader_valid, model, optimizer,loss_fn1,scheduler,scaler):
def train_fn(loader_train,loader_valid, model, optimizer,loss_fn1,scaler): 
     
    train_losses = [] # loss of each batch
    valid_losses = []  # loss of each batch
    
    model.train()

    loop = tqdm(loader_train)
    for batch_idx, (img1,gt1,label) in enumerate(loop):
        img1 = img1.to(device=DEVICE,dtype=torch.float)
        gt1 = gt1.to(device=DEVICE,dtype=torch.float)
    
        # forward
        with torch.cuda.amp.autocast():
            out1 = model(img1)   
            loss1 = loss_fn1(out1, gt1)
           
            
        # backward
        loss=loss1 
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        train_losses.append(loss.item())
    
    loop_v = tqdm(loader_valid)
    model.eval()
    for batch_idx, (img1,gt1,label) in enumerate(loop_v):
        img1 = img1.to(device=DEVICE,dtype=torch.float)
        gt1 = gt1.to(device=DEVICE,dtype=torch.float)
        
        # forward
        with torch.no_grad():
            out1 = model(img1)   
            loss1 = loss_fn1(out1, gt1)
        
        loss=loss1
        loop_v.set_postfix(loss=loss.item())
        valid_losses.append(loss.item())
    model.train()
    
    train_loss_per_epoch = np.average(train_losses)
    valid_loss_per_epoch = np.average(valid_losses)
    ## all epochs
    avg_train_losses.append(train_loss_per_epoch)
    avg_valid_losses.append(valid_loss_per_epoch)
    
    return train_loss_per_epoch,valid_loss_per_epoch
    


def save_checkpoint(state, filename="train1_32.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def check_accuracy(loader, model, device=DEVICE):
    dice_score1=0
    dice_score2=0
    loop = tqdm(loader)
    model.eval()
    with torch.no_grad():
        for batch_idx, (img1,gt1,label) in enumerate(loop):
            img1 = img1.to(device=DEVICE,dtype=torch.float)
            gt1 = gt1.to(device=DEVICE,dtype=torch.float)
            
            p1 = model(img1)  
            
            p1 = (p1 > 0.5) * 1
           
            
            dice_score1 += (2 * (p1 * gt1).sum()) / (
                (p1 + gt1).sum() + 1e-8)
               
                
    print(f"Dice score: {dice_score1/len(loader)}")

    model.train()
    return dice_score1/len(loader)
    
    
             
ALPHA = 0.5
BETA = 0.5
GAMMA = .75

class FocalTverskyLoss(nn.Module):
    def _init_(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self)._init_()

    def forward(self, inputs, targets, smooth=.0001, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky
        
epoch_len = len(str(NUM_EPOCHS))
early_stopping = EarlyStopping(patience=10, verbose=True)
                      
def main():
    model = m_unet_33_1().to(device=DEVICE,dtype=torch.float)
    loss_fn1 =FocalTverskyLoss()
    
        ## Fine Tunnning Part ###
    # optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999),lr=0)
    # weights_paths="/data/home/acw676/seg_//UNET_n.pth.tar"
    # checkpoint = torch.load(weights_paths,map_location=DEVICE)
    # model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
   
   ########################
   
    
    #optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999))
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,step_size_up=2288, base_lr=0.0001, max_lr=0.01,cycle_momentum=False)
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999),lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        #train_loss,valid_loss=train_fn(train_loader,val_loader, model, optimizer, loss_fn1,scheduler,scaler)
        train_loss,valid_loss=train_fn(train_loader,val_loader, model, optimizer, loss_fn1,scaler)
        print_msg = (f'[{epoch:>{epoch_len}}/{NUM_EPOCHS:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        dice_score= check_accuracy(val_loader, model, device=DEVICE)
        avg_valid_DS.append(dice_score.detach().cpu().numpy())
        
        early_stopping(valid_loss, dice_score)
        if early_stopping.early_stop:
            print("Early stopping Reached at  :",epoch)
            break

if __name__ == "__main__":
    main()


avg_train_losses=avg_train_losses
avg_train_losses=avg_train_losses
avg_valid_DS=avg_valid_DS
# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(avg_train_losses)+1),avg_train_losses, label='Training Loss')
plt.plot(range(1,len(avg_valid_losses)+1),avg_valid_losses,label='Validation Loss')
plt.plot(range(1,len(avg_valid_DS)+1),avg_valid_DS,label='Validation DS')

# find position of lowest validation loss
minposs = avg_valid_losses.index(min(avg_valid_losses))+1 
plt.axvline(minposs,linestyle='--', color='r',label='Early Stopping Checkpoint')

font1 = {'size':20}

plt.title("Learning Curve Graph",fontdict = font1)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 1) # consistent scale
plt.xlim(0, len(avg_train_losses)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('train1_32.png', bbox_inches='tight')   