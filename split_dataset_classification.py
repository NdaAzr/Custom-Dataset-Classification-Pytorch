# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:37:18 2019

@author: Neda
"""
from Custom_data_v2 import CustomDataset_classification
import numpy
import glob
import torch

folder_data = glob.glob("D:\\Neda\\Echo_View_Classification\\avi_images\\*\\*.png") # no augmnetation
#numpy.savetxt('distribution_class.csv', numpy.c_[folder_data], fmt=['%s'], comments='', delimiter = ",")                    

 #split these path using a certain percentage
len_data = len(folder_data)
print("count of dataset: ", len_data)

split_1 = int(0.6 * len(folder_data))
split_2 = int(0.8 * len(folder_data))

folder_data.sort()

train_image_paths = folder_data[:split_1]
print("count of train images is: ", len(train_image_paths)) 
numpy.savetxt('im_training_path_1.csv', numpy.c_[train_image_paths], fmt=['%s'], comments='', delimiter = ",")                    


valid_image_paths = folder_data[split_1:split_2]
print("count of validation image is: ", len(valid_image_paths))
numpy.savetxt('im_valid_path_1.csv', numpy.c_[valid_image_paths], fmt=['%s'], comments='', delimiter = ",")     


test_image_paths = folder_data[split_2:]
print("count of test images is: ", len(test_image_paths)) 
numpy.savetxt('im_testing_path_1.csv', numpy.c_[test_image_paths], fmt=['%s'], comments='', delimiter = ",")                    

classes = ['1_PLAX_1_PLAX_full',
  '1_PLAX_2_PLAX_valves',
  '1_PLAX_4_PLAX_TV',
  '2_PSAX_1_PSAX_AV',
  '2_PSAX_2_PSAX_LV',
  '3_Apical_1_MV_LA_IAS',
  '3_Apical_2_A2CH',
  '3_Apical_3_A3CH',
  '3_Apical_5_A5CH',
  '4_A4CH_1_A4CH_LV',
  '4_A4CH_2_A4CH_RV',
  '4_Subcostal_1_Subcostal_heart',
  '4_Subcostal_2_Subcostal_IVC',
  'root_5_Suprasternal',
  'root_6_OTHER']


target = {'1_PLAX_1_PLAX_full': 0,
  '1_PLAX_2_PLAX_valves': 1,
  '1_PLAX_4_PLAX_TV': 2,
  '2_PSAX_1_PSAX_AV': 3,
  '2_PSAX_2_PSAX_LV': 4,
  '3_Apical_1_MV_LA_IAS': 5,
  '3_Apical_2_A2CH': 6,
  '3_Apical_3_A3CH': 7,
  '3_Apical_5_A5CH': 8,
  '4_A4CH_1_A4CH_LV': 9,
  '4_A4CH_2_A4CH_RV': 10,
  '4_Subcostal_1_Subcostal_heart': 11,
  '4_Subcostal_2_Subcostal_IVC': 12,
  'root_5_Suprasternal': 13,
  'root_6_OTHER': 14}


train_dataset = CustomDataset_classification(train_image_paths, classes, target)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=3, shuffle=False, num_workers=2)

valid_dataset = CustomDataset_classification(valid_image_paths, classes, target)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=3, shuffle=False, num_workers=2)


test_dataset = CustomDataset_classification(test_image_paths, classes, target)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)  

dataLoaders = {
        'train': train_loader,
        'valid': valid_loader,
         'test': test_loader,
        }

num_classes = len(train_dataset.classes)
print("number of classes", num_classes) #15
