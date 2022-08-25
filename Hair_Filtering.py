import cv2
import matplotlib.pyplot as plt
import glob


cv_img = []
image_namelist = []
for img in glob.glob("./foto/*.png"):
    image_namelist.append(img)
    n = cv2.imread(img)
    cv_img.append(n)



l = len(cv_img)

#döngü yardımı ile dolaşılır.
for i,image in enumerate(cv_img):
    
    image_name = image_namelist[i]
    image_name = image_name[7:10]
    
    
    img_resize = cv2.resize(image,(256,256))
    plt.subplot(l, 5, (i*5)+1)
    # orjinali grayscale'e dönüştürme
    plt.imshow(cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Origi:'+ image_name)
    #grayscale işlemi
    grayScale = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
    plt.subplot(l, 5, (i*5)+2)
    plt.imshow(grayScale)
    plt.axis('off')
    plt.title('GraySc:'+ image_name)
    
    kernel = cv2.getStructuringElement(1,(17,17))
     
    # blackhat işlemi
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    plt.subplot(l, 5, (i*5)+3)
    plt.imshow(blackhat)
    plt.axis('off')
    plt.title('blackhat:'+ image_name)
    
    # threshold işlemi
    _,thres = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    plt.subplot(l, 5, (i*5)+4)
    plt.imshow(thres)
    plt.axis('off')
    plt.title('thres:'+ image_name)
    
    #Orjinal rengine boyama 
    final_img = cv2.inpaint(img_resize,thres,1,cv2.INPAINT_TELEA)
    plt.subplot(l, 5, (i*5)+5)
    plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('final:'+ image_name)
    cv2.imwrite("./foto2/"+ image_name +".png", final_img)  
    plt.show()
