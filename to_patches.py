import os
import numpy as np
import os 
import glob
    
names=[]
img_files = []
for infile in sorted(glob.glob('/home/akr54/2D data/img/*.npy')):
    img_files.append(infile)
    names.append(infile[24:-4])
sc_files=[]
for n in names:
    sc_files.append('/home/akr54/2D data/sc_gt/'+n+'sc_gt.npy')
b_files=[]
for n in names:
    b_files.append('/home/akr54/2D data/b_gt/'+n+'b_gt.npy')       

path_img='/rds/user/akr54/hpc-work/patches/train/img/'
path_gt='/rds/user/akr54/hpc-work/patches/train/gt/'


def find_(img,sc_gt,b_gt,ID):
    count=0
    sc_count=0
    if np.sum(sc_gt)!=0:
        sc_count=sc_count+1
        for x in range(sc_gt.shape[0]):
           for y in range(sc_gt.shape[1]):
               if b_gt[y,x]==1.0:
                   crop_gt=sc_gt[y-32:y+32,x-32:x+32]
                   if np.sum(crop_gt)>=20:
                       
                       count=count+1
                       if (count % 20 ==0):
                           filename1 = os.path.join(path_gt,ID+'_'+str(count+1))
                           crop_img=img[y-32:y+32,x-32:x+32]
                           filename = os.path.join(path_img,ID+'_'+str(count+1))
                           np.save(filename1, crop_gt)
                           np.save(filename, crop_img)

   
for i in range(2112):
    img=np.load(img_files[i])
    sc_gt=np.load(sc_files[i])
    sc_gt[np.where(sc_gt>0)]=1
    
    b_gt=np.load(b_files[i])
    b_gt[np.where(b_gt>0)]=1
    print(i)
    n_=names[i]
    if np.sum(sc_gt)!=0:
         _=find_(img,sc_gt,b_gt,n_)
          