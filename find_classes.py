# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:02:30 2019
reference :
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html

@author: Neda
"""

import os

def find_classes(dir):   # Finds the class folders in a dataset, dir (string): Root directory path.
        
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


#dir = 'D:/Neda/Echo_View_Classification/avi_images/'
#find_classes(dir)
        
    



