import os
import numpy as np
import SimpleITK as sitk
import torch
from Segmenation import BreastSeg
from Models_3D import ModelBreast
import argparse
from scipy.ndimage import binary_fill_holes, binary_closing, binary_dilation, binary_erosion

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default='1', required=False, help='Run in GPU') 
opt = parser.parse_args()


def refine_mask(mask, structure=np.ones((3, 3))):

    if mask.ndim == 3:  # Process a 3D mask
        refined_slices = []
        for i in range(mask.shape[0]):
            slice_mask = binary_closing(mask[i, :, :], structure=structure)
            refined_slice = binary_fill_holes(slice_mask)
            refined_slices.append(refined_slice)
        refined_mask = np.stack(refined_slices, axis=0)

    return refined_mask.astype(np.uint8)  # Ensure mask is binary (0, 1)

###refining the mask
def refine_mask_2D(mask, structure=np.ones((3, 3))):
    """
    Refine a 2D mask by closing small holes without significantly altering the mask boundary.
    """
    refined_slices = []
    for i in range(mask.shape[0]):  # Assuming the first dimension is the slice dimension
        mask = binary_closing(mask[i, :, :], structure=structure)
        refined_slice = binary_fill_holes(mask)
        refined_slices.append(refined_slice)

    # Stack refined slices back into a 3D array
    refined_mask_3d = np.stack(refined_slices, axis=0)
    return refined_mask_3d.astype(np.uint8)  # Ensure mask is binary (0, 1)

def refine_mask_3D(mask, structure=np.ones((3, 3, 3))):

    # Apply binary closing to close small holes and narrow gaps within the mask
    mask = binary_closing(mask, structure=structure)
    
    # Fill any small holes left within the objects
    mask = binary_fill_holes(mask)

    return mask
#####

def Norm_Zscore(img):
    img= (img-np.mean(img))/np.std(img) 
    return img

def imgnorm(N_I,index1=0.001,index2=0.001):
    N_I = N_I.astype(np.float32)
    I_sort = np.sort(N_I.flatten())
    I_min = I_sort[int(index1*len(I_sort))]
    I_max = I_sort[-int(index2*len(I_sort))]
    
    N_I =1.0*(N_I-I_min)/(I_max-I_min)
    N_I[N_I>1.0]=1.0
    N_I[N_I<0.0]=0.0
    
    return N_I 

def read_image_1(pre_path, post_path):

    I_pre = sitk.ReadImage(pre_path)  
    I_post = sitk.ReadImage(post_path) 
    img_pre = np.array(sitk.GetArrayFromImage(I_pre))
    img_post = np.array(sitk.GetArrayFromImage(I_post))
    
    # Check if the images are multi-channel by examining the array shape
    if img_pre.ndim == 4 and img_post.ndim == 4:
        img_pre = np.mean(img_pre, axis=-1)
        img_post = np.mean(img_post, axis=-1)
    
    img_sub = img_post - img_pre
    
    return img_pre, img_post, img_sub, np.array(I_pre.GetSpacing())

def save_image(img_pre, seg_vol, savename):
    I = sitk.ReadImage(img_pre)  
    Heat_image = sitk.GetImageFromArray(seg_vol, isVector=False) 
    Heat_image.SetSpacing(I.GetSpacing())
    Heat_image.SetOrigin(I.GetOrigin())
    Heat_image.SetDirection(I.GetDirection())
    sitk.WriteImage(Heat_image,savename)  


def process_patient_folder(patient_folder, folder_path, model_breast):
    # pre_path = os.path.join(folder_path, 'Pre.nii')
    # post_path = os.path.join(folder_path, 'Post.nii')
    patient_identifier = patient_folder
    pre_filename = f'{patient_identifier}_pre.nii.gz'
    post_filename = f'{patient_identifier}_post.nii.gz'


    pre_path = os.path.join(folder_path, pre_filename)
    post_path = os.path.join(folder_path, post_filename)
    if not os.path.exists(pre_path) or not os.path.exists(post_path):
        print(f"Missing files for {folder_path}")
        return

    img_pre, img_post, img_sub, spacing = read_image_1(pre_path, post_path)
    # print('Performing image normalization')
    img_pre  = Norm_Zscore(imgnorm(img_pre))
    img_post = Norm_Zscore(imgnorm(img_post))
    img_sub  = Norm_Zscore(imgnorm(img_sub))
    # print(img_pre.shape)
    # print(img_post.shape)
    # print(img_sub.shape)
    breast_mask = BreastSeg(img_post, spacing, model_breast, opt)
    # Save segmentation and subtraction image
    save_image(pre_path, breast_mask, os.path.join(folder_path, 'breast_mask.nii.gz'))
    # save_image(pre_path, img_sub, os.path.join(folder_path, 'img_sub.nii.gz'))
    print(f"Processed and saved results for {folder_path}")

def main(data_dir):

    if opt.cuda:
        model_breast = ModelBreast(1,1).cuda()
    else:
        model_breast = ModelBreast(1,1)
    
    for patient_folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, patient_folder)
        if os.path.isdir(folder_path):
            process_patient_folder(patient_folder,folder_path, model_breast)

if __name__ == "__main__":
    data_dir = "E:/2.Experiments_MRI/5.Localization_slices_classification/dataset"
    main(data_dir)
