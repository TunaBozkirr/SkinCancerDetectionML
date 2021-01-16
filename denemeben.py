import numpy as np
import cv2
import matplotlib.pyplot as plt


def karart(image):   
    img = image
        
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
        
    deger_max = v.max()
        
    thres = 120
    if( deger_max > thres ):
                
        #v[renk] = ( v[renk]*120 / deger_max )
        x = deger_max - 120
        #x = int(deger_max /2)
        
        buyuk_deger_max = v >  thres

        v[buyuk_deger_max] -= x
        
    renk1=  h >= 160
    renk2 = h < 182
    renk = np.logical_and(renk1, renk2)
    
    deger = h[renk]
    deger_max = deger.max()
        
    # temp = v > 120
    thres = 170
    if( deger_max > thres ):                
        #v[renk] = ( v[renk]*120 / deger_max )
        x = deger_max - 18
        #x = int(deger_max /2)
        
        buyuk_deger_max = h >  thres

        h[buyuk_deger_max] -= x
          
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    return img
