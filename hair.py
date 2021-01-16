import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

BASE_PATH = './foto/'
hair_images =['z4','z5','z6']


l = len(hair_images[:3])
print(l)
fig = plt.figure(figsize=(20,30))

for i,image_name in enumerate(hair_images[:3]):
    filename = BASE_PATH + image_name + '.jpeg' 
    
    image = cv2.imread(filename)
    image_resize = cv2.resize(image,(256,256))
    plt.subplot(l, 5, (i*5)+1)
    # original to grayscale
    plt.imshow(cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Original : '+ image_name)
    
    grayScale = cv2.cvtColor(image_resize, cv2.COLOR_RGB2GRAY)
    plt.subplot(l, 5, (i*5)+2)
    plt.imshow(grayScale)
    plt.axis('off')
    plt.title('GrayScale : '+ image_name)
    
    kernel = cv2.getStructuringElement(1,(17,17))
     
    # blackhat
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    plt.subplot(l, 5, (i*5)+3)
    plt.imshow(blackhat)
    plt.axis('off')
    plt.title('blackhat : '+ image_name)
    
    # 
    ret,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    plt.subplot(l, 5, (i*5)+4)
    plt.imshow(threshold)
    plt.axis('off')
    plt.title('threshold : '+ image_name)
    
    # 
    final_image = cv2.inpaint(image_resize,threshold,1,cv2.INPAINT_TELEA)
    plt.subplot(l, 5, (i*5)+5)
    plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('final_image : '+ image_name)
    cv2.imwrite("./"+str(i)+".png", final_image)    
    plt.show() 
