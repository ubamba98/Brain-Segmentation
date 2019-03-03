import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def visualize(PATH, View = "Axial_View",cmap = None):
    '''
    Visualize Image

    Parameters
    ----------
    PATH --- Path to *nii.gz image or image as numpy array
    View --- Sagittal_View or Axial_View or Coronal_View
    cmap

    '''
    plt.ion()
    view = View.lower()
    v = {
        "sagittal_view":0,
        "axial_view":2,
        "coronal_view":1,
    }
    if view not in ["sagittal_view","axial_view","coronal_view"]:
        print("Enter Valid View")
        return
    if(type(PATH) == np.str_ or type(PATH) == str):
        img = nib.load(PATH)
        img1  = img.get_fdata()
    elif(type(PATH) == np.ndarray and len(PATH.shape) == 3):
        img1 = PATH
    else:
        print("ERROR: Input valid type 'PATH'")
        return
    for i in range(img1.shape[v[view]]):
        plt.cla()
        if v[view] == 0:
            plt.imshow(img1[i,:,:], cmap = cmap)
        elif v[view] == 1:
            plt.imshow(img1[:,i,:], cmap = cmap)
        else:
            plt.imshow(img1[:,:,i], cmap = cmap)
        plt.show()
        plt.pause(0.1)
    print(f"Shape of image:{img1.shape}")

def data_train(root = "./"):
    '''
    Return:
    T1path - list of paths to registered Bias Normalized T1
    FLAIRpath - list of paths to registered Bias Normalized FLAIR
    IRpath - list of paths to registered Bias Normalized IR
    segpath - list of paths to segmentations
    '''
    T1path = [root+'training/1/pre/reg_T1.nii.gz',
            root+'training/4/pre/reg_T1.nii.gz',
            root+'training/7/pre/reg_T1.nii.gz',
            root+'training/14/pre/reg_T1.nii.gz',
            root+'training/5/pre/reg_T1.nii.gz',
            root+'training/070/pre/reg_T1.nii.gz']
    FLAIRpath = [root+'training/1/pre/FLAIR.nii.gz',
            root+'training/4/pre/FLAIR.nii.gz',
            root+'training/7/pre/FLAIR.nii.gz',
            root+'training/14/pre/FLAIR.nii.gz',
            root+'training/5/pre/FLAIR.nii.gz',
            root+'training/070/pre/FLAIR.nii.gz']
    IRpath = [root+'training/1/pre/reg_IR.nii.gz',
            root+'training/4/pre/reg_IR.nii.gz',
            root+'training/7/pre/reg_IR.nii.gz',
            root+'training/14/pre/reg_IR.nii.gz',
            root+'training/5/pre/reg_IR.nii.gz',
            root+'training/070/pre/reg_IR.nii.gz']
    segpath = [root+'training/1/segm.nii.gz',
             root+'training/4/segm.nii.gz',
             root+'training/7/segm.nii.gz',
             root+'training/14/segm.nii.gz',
             root+'training/5/segm.nii.gz',
             root+'training/070/segm.nii.gz']
    return T1path, FLAIRpath, IRpath, segpath

def data_val(root = "./"):
    '''
    Return:
    T1_val - Path to registered Bias Normalized T1
    FLAIR_val - Path to registered Bias Normalized FLAIR
    IR_val - Path to registered Bias Normalized IR
    segm_val - Path to segmentations
    '''
    T1_val = root+'training/148/pre/reg_T1.nii.gz'
    FLAIR_val = root+'training/148/pre/FLAIR.nii.gz'
    IR_val = root+'training/148/pre/reg_IR.nii.gz'
    segm_val = root+'training/148/segm.nii.gz'
    return T1_val, FLAIR_val, IR_val, segm_val
