import os
import numpy as np
import torch
from skimage import io
from torch.utils import data
import utils.transform as transform
import matplotlib.pyplot as plt
from skimage.transform import rescale
from skimage.color import rgb2gray
from torchvision.transforms import functional as F
# from osgeo import gdal_array
import cv2
# from regionup import area_grow
num_classes = 6
ST_COLORMAP = [[255,255,255], [128,0,0], [128,128,128], [0,128,0], [0,255,0], [0,0,255]]
ST_CLASSES = ['unchanged', 'Artificial surfaces', 'Agricultural areas', 'Forests', 'Wetlands', 'Water']

MEAN_A = np.array([113.40, 114.08, 116.45])
STD_A  = np.array([48.30,  46.27,  48.14])
MEAN_B = np.array([111.07, 114.04, 118.18])
STD_B  = np.array([49.41,  47.01,  47.94])

root = '/'

colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(ST_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

def Colorls2Index(ColorLabels):
    IndexLabels = []
    for i, data in enumerate(ColorLabels):
        IndexMap = Color2Index(data)
        IndexLabels.append(IndexMap)
    return IndexLabels

def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    #IndexMap = 2*(IndexMap > 1) + 1 * (IndexMap <= 1)
    IndexMap = IndexMap * (IndexMap < num_classes)
    return IndexMap

def Index2Color(pred):
    colormap = np.asarray(ST_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]

def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0

def normalize_image(im, time='A'):
    assert time in ['A', 'B']
    if time=='A':
        im = (im - MEAN_A) / STD_A
    else:
        im = (im - MEAN_B) / STD_B
    return im

def normalize_images(imgs, time='A'):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im, time)
    return imgs

# def read_RSimages(mode, rescale=False):
#     #assert mode in ['train', 'val', 'test']
#     img_A_dir = os.path.join(root, mode, 'im1')
#     img_B_dir = os.path.join(root, mode, 'im2')
#     label_A_dir = os.path.join(root, mode, 'label1')
#     label_B_dir = os.path.join(root, mode, 'label2')
#     label_A_grow_dir = os.path.join(root, mode, 'out_label1')
#     label_B_grow_dir = os.path.join(root, mode, 'out_label2')
#     # To use rgb labels:
#     #label_A_dir = os.path.join(root, mode, 'label1_rgb')
#     #label_B_dir = os.path.join(root, mode, 'label2_rgb')
    
#     data_list = os.listdir(img_A_dir)
#     imgs_list_A, imgs_list_B, labels_A, labels_B,labels_A_grow,labels_B_grow = [], [], [], [], [], []
#     count = 0
#     for it in data_list:
#         # print(it)
#         if (it[-4:]=='.png'):
#             img_A_path = os.path.join(img_A_dir, it)
#             img_B_path = os.path.join(img_B_dir, it)
#             label_A_path = os.path.join(label_A_dir, it)
#             label_B_path = os.path.join(label_B_dir, it)
#             label_A_grow_path = os.path.join(label_A_grow_dir, it)
#             label_B_grow_path = os.path.join(label_B_grow_dir, it)
#             imgs_list_A.append(img_A_path)
#             imgs_list_B.append(img_B_path)
            
#             label_A = io.imread(label_A_path)
#             label_B = io.imread(label_B_path)
#             label_A_grow = io.imread(label_A_grow_path)
#             label_B_grow = io.imread(label_B_grow_path)
#             #for rgb labels:
#             #label_A = Color2Index(label_A)
#             #label_B = Color2Index(label_B)
#             labels_A.append(label_A)
#             labels_B.append(label_B)
#             labels_A_grow.append(label_A_grow)
#             labels_B_grow.append(label_B_grow)
#         count+=1
#         if not count%500: print('%d/%d images loaded.'%(count, len(data_list)))
    
#     print(labels_A[0].shape)
#     print(str(len(imgs_list_A)) + ' ' + mode + ' images' + ' loaded.')
    
#     return imgs_list_A, imgs_list_B, labels_A, labels_B,labels_A_grow,labels_B_grow

# class Data(data.Dataset):
#     def __init__(self, mode, random_flip = False):
#         self.random_flip = random_flip
#         self.imgs_list_A, self.imgs_list_B, self.labels_A, self.labels_B ,self.labels_A_grow,self.labels_B_grow= read_RSimages(mode)
    
#     def get_mask_name(self, idx):
#         mask_name = os.path.split(self.imgs_list_A[idx])[-1]
#         return mask_name

#     def __getitem__(self, idx):
#         # img_A = cv2.imread(self.imgs_list_A[idx])
#         # img_B = cv2.imread(self.imgs_list_B[idx])
#         # img_a = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY) 
#         # img_b = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)
#         # label_A = self.labels_A[idx]
#         # label_B = self.labels_B[idx]
#         # label_A_grow = area_grow(img_a,label_A,t=5)
#         # label_B_grow = area_grow(img_b,label_B,t=5)
#         # img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB) 
#         # img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB) 
#         # img_A = normalize_image(img_A, 'A')
#         # img_B = normalize_image(img_B, 'B')
#         img_A = io.imread(self.imgs_list_A[idx])
#         img_A = normalize_image(img_A, 'A')
#         img_B = io.imread(self.imgs_list_B[idx])
#         img_B = normalize_image(img_B, 'B')
#         label_A = self.labels_A[idx]
#         label_B = self.labels_B[idx]
#         label_A_grow = self.labels_A_grow[idx]
#         label_B_grow = self.labels_B_grow[idx]
#         if self.random_flip:
#             img_A, img_B, label_A, label_B ,label_A_grow,label_B_grow= transform.rand_rot90_flip_MCD(img_A, img_B, label_A, label_B,label_A_grow,label_B_grow)
#         return F.to_tensor(img_A), F.to_tensor(img_B), torch.from_numpy(label_A), torch.from_numpy(label_B),torch.from_numpy(label_A_grow),torch.from_numpy(label_B_grow)

#     def __len__(self):
#         return len(self.imgs_list_A)
def read_RSimages(mode, rescale=False):
    #assert mode in ['train', 'val', 'test']
    img_A_dir = os.path.join(root, mode, 'A')
    img_B_dir = os.path.join(root, mode, 'B')
    label_A_dir = os.path.join(root, mode, 'label1')
    label_B_dir = os.path.join(root, mode, 'label2')
    # To use rgb labels:
    #label_A_dir = os.path.join(root, mode, 'label1_rgb')
    #label_B_dir = os.path.join(root, mode, 'label2_rgb')
    
    data_list = sorted(os.listdir(img_A_dir))
    imgs_list_A, imgs_list_B, labels_A, labels_B = [], [], [], []
    count = 0
    for it in data_list:
        # print(it)
        if (it[-4:]=='.png'):
            img_A_path = os.path.join(img_A_dir, it)
            img_B_path = os.path.join(img_B_dir, it)
            label_A_path = os.path.join(label_A_dir, it)
            label_B_path = os.path.join(label_B_dir, it)
            
            imgs_list_A.append(img_A_path)
            imgs_list_B.append(img_B_path)
            
            label_A = np.array(io.imread(label_A_path),dtype = np.float32)
            label_B = np.array(io.imread(label_B_path),dtype = np.float32)
            #for rgb labels:
            #label_A = Color2Index(label_A)
            #label_B = Color2Index(label_B)
            labels_A.append(label_A)
            labels_B.append(label_B)
        count+=1
        if not count%500: print('%d/%d images loaded.'%(count, len(data_list)))
    
    print(labels_A[0].shape)
    print(str(len(imgs_list_A)) + ' ' + mode + ' images' + ' loaded.')
    
    return imgs_list_A, imgs_list_B, labels_A, labels_B

class Data(data.Dataset):
    def __init__(self, mode, random_flip = False):
        self.random_flip = random_flip
        self.imgs_list_A, self.imgs_list_B, self.labels_A, self.labels_B = read_RSimages(mode)
    
    def get_mask_name(self, idx):
        mask_name = os.path.split(self.imgs_list_A[idx])[-1]
        return mask_name

    def __getitem__(self, idx):
        img_A = io.imread(self.imgs_list_A[idx])
        img_A = normalize_image(img_A, 'A')
        img_B = io.imread(self.imgs_list_B[idx])
        img_B = normalize_image(img_B, 'B')
        label_A = self.labels_A[idx]
        label_B = self.labels_B[idx]
        if self.random_flip:
            img_A, img_B, label_A, label_B = transform.rand_rot90_flip_MCD(img_A, img_B, label_A, label_B)
        return F.to_tensor(img_A), F.to_tensor(img_B), torch.from_numpy(label_A), torch.from_numpy(label_B)

    def __len__(self):
        return len(self.imgs_list_A)-1
def read_RSimages1(mode, rescale=False):
    #assert mode in ['train', 'val', 'test']
    img_A_dir = os.path.join(root, mode, 'A')
    img_B_dir = os.path.join(root, mode, 'B')
    label_A_dir = os.path.join(root, mode, 'label1')
    label_B_dir = os.path.join(root, mode, 'label2')
    label_dir = os.path.join(root, mode, 'label')
    # To use rgb labels:
    #label_A_dir = os.path.join(root, mode, 'label1_rgb')
    #label_B_dir = os.path.join(root, mode, 'label2_rgb')
    
    data_list = os.listdir(img_A_dir)
    imgs_list_A, imgs_list_B, labels_A, labels_B ,labels= [], [], [], [], []
    count = 0
    for it in data_list:
        # print(it)
        if (it[-4:]=='.png'):
            img_A_path = os.path.join(img_A_dir, it)
            img_B_path = os.path.join(img_B_dir, it)
            label_A_path = os.path.join(label_A_dir, it)
            label_B_path = os.path.join(label_B_dir, it)
            label_path = os.path.join(label_dir, it)
            
            imgs_list_A.append(img_A_path)
            imgs_list_B.append(img_B_path)
            
            # label_A = io.imread(label_A_path)
            # label_B = io.imread(label_B_path)
            label_A = np.array(io.imread(label_A_path),dtype = np.float32)
            label_B = np.array(io.imread(label_B_path),dtype = np.float32)
            label = np.array(io.imread(label_path),dtype = np.float32)

            #for rgb labels:
            #label_A = Color2Index(label_A)
            #label_B = Color2Index(label_B)
            labels_A.append(label_A)
            labels_B.append(label_B)
            labels.append(label)
        count+=1
        if not count%500: print('%d/%d images loaded.'%(count, len(data_list)))
    
    print(labels_A[0].shape)
    print(str(len(imgs_list_A)) + ' ' + mode + ' images' + ' loaded.')
    
    return imgs_list_A, imgs_list_B, labels_A, labels_B, labels

class Data1(data.Dataset):
    def __init__(self, mode, random_flip = False):
        self.random_flip = random_flip
        self.imgs_list_A, self.imgs_list_B, self.labels_A, self.labels_B, self.labels = read_RSimages1(mode)
    
    def get_mask_name(self, idx):
        mask_name = os.path.split(self.imgs_list_A[idx])[-1]
        return mask_name

    def __getitem__(self, idx):
        img_A = io.imread(self.imgs_list_A[idx])
        img_A = normalize_image(img_A, 'A')
        img_B = io.imread(self.imgs_list_B[idx])
        img_B = normalize_image(img_B, 'B')
        label_A = self.labels_A[idx]
        label_B = self.labels_B[idx]
        label = self.labels[idx]
        if self.random_flip:
            img_A, img_B, label_A, label_B, label = transform.rand_rot90_flip_MCD_mask(img_A, img_B, label_A, label_B,label)
        return F.to_tensor(img_A), F.to_tensor(img_B), torch.from_numpy(label_A), torch.from_numpy(label_B), torch.from_numpy(label)

    def __len__(self):
        return len(self.imgs_list_A)
class Data_test(data.Dataset):
    def __init__(self, test_dir):
        self.imgs_A = []
        self.imgs_B = []
        self.mask_name_list = []
        imgA_dir = os.path.join(test_dir, 'A')
        imgB_dir = os.path.join(test_dir, 'B')
        data_list = sorted(os.listdir(imgA_dir))
        for it in data_list:
            if (it[-4:]=='.png'):
                img_A_path = os.path.join(imgA_dir, it)
                img_B_path = os.path.join(imgB_dir, it)
                self.imgs_A.append(io.imread(img_A_path))
                self.imgs_B.append(io.imread(img_B_path))
                self.mask_name_list.append(it)
        self.len = len(self.imgs_A)

    def get_mask_name(self, idx):
        return self.mask_name_list[idx]

    def __getitem__(self, idx):
        img_A = self.imgs_A[idx]
        img_B = self.imgs_B[idx]
        img_A = normalize_image(img_A, 'A')
        img_B = normalize_image(img_B, 'B')
        return F.to_tensor(img_A), F.to_tensor(img_B)

    def __len__(self):
        return self.len

class Data_test_eval(data.Dataset):
    def __init__(self, test_dir):
        self.imgs_A_label = []
        self.imgs_B_label = []
        self.mask_name_list = []
        imgA_dir = os.path.join(test_dir, 'label1')
        imgB_dir = os.path.join(test_dir, 'label2')
        data_list = sorted(os.listdir(imgA_dir))
        for it in data_list:
            if (it[-4:]=='.png'):
                img_A_path = os.path.join(imgA_dir, it)
                img_B_path = os.path.join(imgB_dir, it)
                self.imgs_A_label.append(io.imread(img_A_path))
                self.imgs_B_label.append(io.imread(img_B_path))
                self.mask_name_list.append(it)
        self.len = len(self.imgs_A_label)

    def get_mask_name(self, idx):
        return self.mask_name_list[idx]

    def __getitem__(self, idx):
        label1 = self.imgs_A_label[idx]
        label2 = self.imgs_B_label[idx]
        return torch.from_numpy(label1), torch.from_numpy(label2)

    def __len__(self):
        return self.len
class Data_test_eval1(data.Dataset):
    def __init__(self, test_dir):
        self.imgs_A_label = []
        self.imgs_B_label = []
        self.imgs_label = []
        self.mask_name_list = []
        imgA_dir = os.path.join(test_dir, 'label1')
        imgB_dir = os.path.join(test_dir, 'label2')
        imglabel_dir = os.path.join(test_dir, 'label')
        data_list = sorted(os.listdir(imgA_dir))
        for it in data_list:
            if (it[-4:]=='.png'):
                img_A_path = os.path.join(imgA_dir, it)
                img_B_path = os.path.join(imgB_dir, it)
                img_label_path = os.path.join(imglabel_dir, it)
                self.imgs_A_label.append(np.array(io.imread(img_A_path),dtype = np.float32))
                self.imgs_B_label.append(np.array(io.imread(img_B_path),dtype = np.float32))
                self.imgs_label.append(np.array(io.imread(img_label_path),dtype = np.float32))
                self.mask_name_list.append(it)
        self.len = len(self.imgs_A_label)

    def get_mask_name(self, idx):
        return self.mask_name_list[idx]

    def __getitem__(self, idx):
        label1 = self.imgs_A_label[idx]
        label2 = self.imgs_B_label[idx]
        label = self.imgs_label[idx]
        return torch.from_numpy(label1), torch.from_numpy(label2), torch.from_numpy(label)

    def __len__(self):
        return self.len