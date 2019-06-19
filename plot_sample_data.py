# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:17:28 2019

@author: Neda
"""

from split_dataset_classification import train_loader, classes, dataLoaders
import numpy
import csv
import torchvision
import matplotlib.pyplot as plt

def visualize():
    # functions to show an image
    def imshow(img):
        npimg = img.numpy()
        plt.imshow(numpy.transpose(npimg, (1, 2, 0)))
        plt.show()
        
    data_iterator = iter(train_loader)
    t_image, target, image_paths = data_iterator.next()
    print(t_image.shape)
    #print(t_image)
    print(target)
    #print(image_paths)
    
    plt.figure()
    imshow(torchvision.utils.make_grid(t_image))
    print(' '.join('%5s' % classes[j] for j in range(3)))
    print(' '.join('%5s' % image_paths[j] for j in range(3)))    
    # =============================================================================   
    data_iterator = iter(train_loader)
    t_image, target, image_paths = data_iterator.next()
            
    #plot images to visualize the data
    rows = 1
    columns = 3
    fig=plt.figure()
    for i in range(3):
        
        fig.add_subplot(rows, columns, i+1)
        plt.title(classes[i])
        img = t_image[i]      
        img = torchvision.transforms.ToPILImage()(img)
        plt.imshow(img, cmap='gray')
    plt.show()


if __name__=='__main__':
    visualize()