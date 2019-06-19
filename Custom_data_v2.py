# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:31:31 2019
https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html
@author: Neda
"""
from torch.utils.data.dataset import Dataset  
import torchvision.transforms as transforms
from PIL import Image
from find_classes import find_classes
from make_dataset import make_dataset


class CustomDataset_classification(Dataset):  
    
    
    def __init__(self, image_paths, classes, class_to_id):  
        
        self.image_paths = image_paths
        self.transforms = transforms.ToTensor() 
        classes, class_to_idx = find_classes('D:/Neda/Echo_View_Classification/avi_images/')
        samples = make_dataset('D:/Neda/Echo_View_Classification/avi_images/', class_to_idx)

        self.classes = classes
        self.class_to_idx = class_to_idx   
        self.samples = samples
        self.targets = [s[1] for s in samples]
    
    def __getitem__(self, index):

        image = Image.open(self.image_paths[index])
        t_image = image.convert('L')
        t_image = self.transforms(t_image) 
        
        path, target = self.samples[index]
                               
        return t_image, target, self.image_paths[index]
    
    def __len__(self): 

        return len(self.samples)
     

