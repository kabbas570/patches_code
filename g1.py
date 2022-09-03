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



# batch_size=2


# val_imgs=r'C:\Users\Abbas Khan\Desktop\patches\img'

# val_masks=r'C:\Users\Abbas Khan\Desktop\patches\gt'

# test_loader=Data_Loader(val_imgs,val_masks,batch_size)
# print(len(test_loader))

# a=iter(test_loader)
# a1=next(a)
# img=a1[0][0,1,:,:].numpy()

# gt=a1[1][0,0,:,:].numpy()
