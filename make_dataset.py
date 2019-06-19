# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:49:11 2019

@author: Neda
"""
import os
import os.path

def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])
                images.append(item)
    
    return images

class_to_idx = {'1_PLAX_1_PLAX_full': 0,
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

direct = 'D:/Neda/Echo_View_Classification/avi_images/'
make_dataset(direct, class_to_idx)