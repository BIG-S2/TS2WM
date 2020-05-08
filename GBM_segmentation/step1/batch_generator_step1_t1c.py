import nibabel as nib
import numpy as np
from keras.utils import to_categorical
import os
from nilearn.image import new_img_like, resample_to_img
import nibabel as nib

# step1
def scale_image(image, scale_factor):
    scale_factor = np.asarray(scale_factor)
    new_affine = np.copy(image.affine)
    new_affine[:3, :3] = image.affine[:3, :3] * scale_factor
    new_affine[:, 3][:3] = image.affine[:, 3][:3] + (image.shape * np.diag(image.affine)[:3] * (1 - scale_factor)) / 2
    return new_img_like(image, data=image.get_data(), affine=new_affine)

#truth_data=get_random_eraser(T1)
def cut_edge(data, keep_margin):
    '''
    function that cuts zero edge
    '''
    D, H, W = data.shape
    D_s, D_e = 0, D - 1
    H_s, H_e = 0, H - 1
    W_s, W_e = 0, W - 1

    while D_s < D:
        if data[D_s].sum() != 0:
            break
        D_s += 1
    while D_e > D_s:
        if data[D_e].sum() != 0:
            break
        D_e -= 1
    while H_s < H:
        if data[:, H_s].sum() != 0:
            break
        H_s += 1
    while H_e > H_s:
        if data[:, H_e].sum() != 0:
            break
        H_e -= 1
    while W_s < W:
        if data[:, :, W_s].sum() != 0:
            break
        W_s += 1
    while W_e > W_s:
        if data[:, :, W_e].sum() != 0:
            break
        W_e -= 1

    if keep_margin != 0:
        D_s = max(0, D_s - keep_margin)
        D_e = min(D - 1, D_e + keep_margin)
        H_s = max(0, H_s - keep_margin)
        H_e = min(H - 1, H_e + keep_margin)
        W_s = max(0, W_s - 4)
        W_e = min(W - 1, W_e + 4)
    return int(D_s), int(D_e), int(H_s), int(H_e), int(W_s), int(W_e)


def training_generator(temp):
    patch_size=64
    patch=16
    margin = int(patch_size/2)
    patch_num=12
    patch_num1=10
    x_data=np.zeros([40000,64,64,16,3])
    y_data=np.zeros([40000,64,64,16])
    time=-1
    tr_X1='/HGG/'
    lst = os.listdir(tr_X1)
    lst.sort()
    tr_X2='/LGG/'
    lst1 = os.listdir(tr_X2)
    lst1.sort()
    for sub in range (0,220):
        
        if sub<160:
        
            n=lst[int(temp[sub])-1]
            T1c=nib.load('/HGG/'+n+'/T1c.nii').get_data()
            Mask=nib.load('/HGG/'+n+'/seg.nii').get_data() # ground true segmentation

        else:
            n=lst1[int(temp[sub])-1]
            T1c=nib.load('/LGG/'+n+'/T1c.nii').get_data()
            Mask=nib.load('/LGG/'+n+'/seg.nii').get_data()

        Mask[Mask==4]=3
        for flip_num in range (6):  
            T1_temp=np.copy(T1c)
            label_vols=np.copy(Mask)
            if flip_num-2>0:
                    T1_temp=np.flip(T1_temp,axis=flip_num-3)
                    label_vols=np.flip(label_vols,axis=flip_num-3)
                
            if flip_num==2:
                    noise = np.random.normal(0,1,T1_temp.shape)
                    T1_temp = np.add(T1_temp, noise)
                    
            min_D_s, max_D_e, min_H_s, max_H_e, min_W_s, max_W_e = cut_edge(T1_temp, margin)
            inputs_T1 = T1_temp[min_D_s:max_D_e + 1, min_H_s: max_H_e + 1, min_W_s:max_W_e + 1, None]
            labels = label_vols[min_D_s:max_D_e + 1, min_H_s: max_H_e + 1, min_W_s:max_W_e + 1]
            
             
            inputs = np.concatenate((inputs_T1,inputs_T1, inputs_T1), axis=3)
            if flip_num==1:
                inputs=np.transpose(inputs,(1,0,2,3))
                labels=np.transpose(labels,(1,0,2))
                
            rnd0 = np.random.randint(0,labels.shape[0]-patch_size,size=patch_num)   
            rnd1 = np.random.randint(0,labels.shape[1]-patch_size,size=patch_num)
            rnd2 = np.random.randint(0,labels.shape[2]-patch,size=patch_num)    
        
             
            for ind in range(patch_num):
                if len(np.unique(labels[rnd0[ind]:(rnd0[ind]+patch_size), rnd1[ind]:(rnd1[ind]+patch_size), rnd2[ind]:(rnd2[ind]+patch)]))<2:
                    time+=1
                    x_data[time]=inputs[rnd0[ind]:(rnd0[ind]+patch_size), rnd1[ind]:(rnd1[ind]+patch_size), rnd2[ind]:(rnd2[ind]+patch),:3]
                    y_data[time]=labels[rnd0[ind]:(rnd0[ind]+patch_size), rnd1[ind]:(rnd1[ind]+patch_size), rnd2[ind]:(rnd2[ind]+patch)]

            min_D_s, max_D_e, min_H_s, max_H_e, min_W_s, max_W_e = cut_edge(label_vols, margin)
            inputs_T1 = T1_temp[min_D_s:max_D_e + 1, min_H_s: max_H_e + 1, min_W_s:max_W_e + 1, None]
            labels = label_vols[min_D_s:max_D_e + 1, min_H_s: max_H_e + 1, min_W_s:max_W_e + 1]

             
            inputs = np.concatenate((inputs_T1,inputs_T1, inputs_T1), axis=3)
            if flip_num==1:
                inputs=np.transpose(inputs,(1,0,2,3))
                labels=np.transpose(labels,(1,0,2))
                
            rnd0 = np.random.randint(0,labels.shape[0]-patch_size,size=patch_num1)   
            rnd1 = np.random.randint(0,labels.shape[1]-patch_size,size=patch_num1)
            rnd2 = np.random.randint(0,labels.shape[2]-patch,size=patch_num1)    
        
             
            for ind in range(patch_num1):
                time+=1
                x_data[time]=inputs[rnd0[ind]:(rnd0[ind]+patch_size), rnd1[ind]:(rnd1[ind]+patch_size), rnd2[ind]:(rnd2[ind]+patch),:3]
                y_data[time]=labels[rnd0[ind]:(rnd0[ind]+patch_size), rnd1[ind]:(rnd1[ind]+patch_size), rnd2[ind]:(rnd2[ind]+patch)]
    #scale

    # for sub in range (0,220):
    #         if sub<160:
    #
    #             n=lst[int(temp[sub])-1]
    #             T1c=nib.load('/HGG/'+n+'/T1c.nii')
    #             T1c = resample_to_img(scale_image(T1c,0.75),
    #                                  T1c, interpolation="nearest").get_data()
    #             Mask=nib.load('/HGG/'+n+'/seg.nii')
    #             Mask = resample_to_img(scale_image(Mask,0.75),
    #                                  Mask, interpolation="nearest").get_data()
    #         else:
    #             n=lst1[int(temp[sub])-1]
    #             T1c=nib.load('/LGG/'+n+'/T1c.nii')
    #             T1c = resample_to_img(scale_image(T1c,0.75),
    #                                  T1c, interpolation="nearest").get_data()
    #             Mask=nib.load('/LGG/'+n+'/seg.nii')
    #             Mask = resample_to_img(scale_image(Mask,0.75),
    #                                  Mask, interpolation="nearest").get_data()
    #
    #         Mask[Mask==4]=3
    #         for flip_num in range (4):
    #             T1_temp=np.copy(T1c)
    #             label_vols=np.copy(Mask)
    #             if flip_num>0:
    #                     T1_temp=np.flip(T1_temp,axis=flip_num-1)
    #                     label_vols=np.flip(label_vols,axis=flip_num-1)
    #
    #
    #             min_D_s, max_D_e, min_H_s, max_H_e, min_W_s, max_W_e = cut_edge(T1_temp, margin)
    #             inputs_T1 = T1_temp[min_D_s:max_D_e + 1, min_H_s: max_H_e + 1, min_W_s:max_W_e + 1, None]
    #             labels = label_vols[min_D_s:max_D_e + 1, min_H_s: max_H_e + 1, min_W_s:max_W_e + 1]
    #
    #
    #             inputs = np.concatenate((inputs_T1,inputs_T1, inputs_T1), axis=3)
    #
    #             rnd0 = np.random.randint(0,labels.shape[0]-patch_size,size=patch_num)
    #             rnd1 = np.random.randint(0,labels.shape[1]-patch_size,size=patch_num)
    #             rnd2 = np.random.randint(0,labels.shape[2]-patch,size=patch_num)
    #
    #
    #             for ind in range(patch_num):
    #                 if len(np.unique(labels[rnd0[ind]:(rnd0[ind]+patch_size), rnd1[ind]:(rnd1[ind]+patch_size), rnd2[ind]:(rnd2[ind]+patch)]))<2:
    #                     time+=1
    #                     x_data[time]=inputs[rnd0[ind]:(rnd0[ind]+patch_size), rnd1[ind]:(rnd1[ind]+patch_size), rnd2[ind]:(rnd2[ind]+patch),:3]
    #                     y_data[time]=labels[rnd0[ind]:(rnd0[ind]+patch_size), rnd1[ind]:(rnd1[ind]+patch_size), rnd2[ind]:(rnd2[ind]+patch)]
    #
    #             min_D_s, max_D_e, min_H_s, max_H_e, min_W_s, max_W_e = cut_edge(label_vols, margin)
    #             inputs_T1 = T1_temp[min_D_s:max_D_e + 1, min_H_s: max_H_e + 1, min_W_s:max_W_e + 1, None]
    #             labels = label_vols[min_D_s:max_D_e + 1, min_H_s: max_H_e + 1, min_W_s:max_W_e + 1]
    #
    #
    #             inputs = np.concatenate((inputs_T1,inputs_T1, inputs_T1), axis=3)
    #
    #             rnd0 = np.random.randint(0,labels.shape[0]-patch_size,size=patch_num1)
    #             rnd1 = np.random.randint(0,labels.shape[1]-patch_size,size=patch_num1)
    #             rnd2 = np.random.randint(0,labels.shape[2]-patch,size=patch_num1)
    #
    #
    #             for ind in range(patch_num1):
    #                 time+=1
    #                 x_data[time]=inputs[rnd0[ind]:(rnd0[ind]+patch_size), rnd1[ind]:(rnd1[ind]+patch_size), rnd2[ind]:(rnd2[ind]+patch),:3]
    #                 y_data[time]=labels[rnd0[ind]:(rnd0[ind]+patch_size), rnd1[ind]:(rnd1[ind]+patch_size), rnd2[ind]:(rnd2[ind]+patch)]

    x_data=x_data[0:time,:]
    y_data=y_data[0:time,:]
    y_data=to_categorical(y_data,4)
    return x_data,y_data

def validate_generator(temp1):
    patch_size=64
    patch=16
    margin = int(patch_size/2)
    patch_num=60
    patch_num1=100
    x_val=np.zeros([20000,64,64,16,3])
    y_val=np.zeros([20000,64,64,16])
    time=-1
    tr_X1='/HGG/'
    lst = os.listdir(tr_X1)
    lst.sort()
    tr_X2='/LGG/'
    lst1 = os.listdir(tr_X2)
    lst1.sort()
    for sub in range (0,55):
        if sub<160:
            n=lst[int(temp1[sub])-1]
            T1c=nib.load('/HGG/'+n+'/T1c.nii').get_data()
            Mask=nib.load('/HGG/'+n+'/seg.nii').get_data()

        else:
            n=lst1[int(temp1[sub])-1]
            T1c=nib.load('/LGG/'+n+'/T1c.nii').get_data()
            Mask=nib.load('/scratch/lzhong4/train/LGG/'+n+'/seg.nii').get_data()

        Mask[Mask==4]=3 
        T1_temp=np.copy(T1c)
        label_vols=np.copy(Mask)
        
        min_D_s, max_D_e, min_H_s, max_H_e, min_W_s, max_W_e = cut_edge(T1_temp, margin)
        inputs_T1 = T1_temp[min_D_s:max_D_e + 1, min_H_s: max_H_e + 1, min_W_s:max_W_e + 1, None]
        labels = label_vols[min_D_s:max_D_e + 1, min_H_s: max_H_e + 1, min_W_s:max_W_e + 1]
        
         
        inputs = np.concatenate((inputs_T1,inputs_T1, inputs_T1), axis=3)
            
        rnd0 = np.random.randint(0,labels.shape[0]-patch_size,size=patch_num)   
        rnd1 = np.random.randint(0,labels.shape[1]-patch_size,size=patch_num)
        rnd2 = np.random.randint(0,labels.shape[2]-patch,size=patch_num)    
    
         
        for ind in range(patch_num):
            if len(np.unique(labels[rnd0[ind]:(rnd0[ind]+patch_size), rnd1[ind]:(rnd1[ind]+patch_size), rnd2[ind]:(rnd2[ind]+patch)]))<2:
                time+=1
                x_val[time]=inputs[rnd0[ind]:(rnd0[ind]+patch_size), rnd1[ind]:(rnd1[ind]+patch_size), rnd2[ind]:(rnd2[ind]+patch),:3]
                y_val[time]=labels[rnd0[ind]:(rnd0[ind]+patch_size), rnd1[ind]:(rnd1[ind]+patch_size), rnd2[ind]:(rnd2[ind]+patch)]

        min_D_s, max_D_e, min_H_s, max_H_e, min_W_s, max_W_e = cut_edge(label_vols, margin)
        inputs_T1 = T1_temp[min_D_s:max_D_e + 1, min_H_s: max_H_e + 1, min_W_s:max_W_e + 1, None]
        labels = label_vols[min_D_s:max_D_e + 1, min_H_s: max_H_e + 1, min_W_s:max_W_e + 1]

         
        inputs = np.concatenate((inputs_T1,inputs_T1, inputs_T1), axis=3)
            
        rnd0 = np.random.randint(0,labels.shape[0]-patch_size,size=patch_num1)   
        rnd1 = np.random.randint(0,labels.shape[1]-patch_size,size=patch_num1)
        rnd2 = np.random.randint(0,labels.shape[2]-patch,size=patch_num1)    
    
         
        for ind in range(patch_num1):
            time+=1
            x_val[time]=inputs[rnd0[ind]:(rnd0[ind]+patch_size), rnd1[ind]:(rnd1[ind]+patch_size), rnd2[ind]:(rnd2[ind]+patch),:3]
            y_val[time]=labels[rnd0[ind]:(rnd0[ind]+patch_size), rnd1[ind]:(rnd1[ind]+patch_size), rnd2[ind]:(rnd2[ind]+patch)]

    x_val=x_val[0:time,:]
    y_val=y_val[0:time,:]
    y_val=to_categorical(y_val,4)
    return x_val,y_val
