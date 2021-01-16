import numpy as np
import cv2
import matplotlib.pyplot as plt
from denemeben import karart
import math


image_as = cv2.imread("2.png")

image = karart(image_as)

# BEN BÖLÜTLEME  
boundaries = [([0, 100, 19], [17, 200, 120])]  #(ilerde)2. renk aralığı verilerek 1. olmazsa 2.si denenecek
            
#  black,([0, 0, 0],[180, 255, 30])
#  sarı dahil [([0, 150, 25], [20, 200, 200])]
#  çok renkli ben için ideal ([0, 115, 19], [17, 200, 120])

converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)




for (lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")
     
    # find the colors within the specified boundaries and apply
    # the mask
    mask  = cv2.inRange(converted, lower, upper)
    plt.imshow(mask)
    plt.show()
    
    output = cv2.bitwise_and(image, image, mask = mask )
    #plt.imshow(output)
    #plt.show()

    contours, _ = cv2.findContours(mask , cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    
    #lezyon etrafına çember çizdirme
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    print("daire merkez",center)
    radius = int(radius)
    imageb = cv2.circle(image,center,radius,(0,255,0),2)
    
    
    # En büyük konturun merkez koorinatlarını bulma
    for c in contours:        
        if cv2.contourArea(c) == (cv2.contourArea(cnt)) :
            
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
                     
            
        	# draw the contour and center of the shape on the image
            cv2.circle(image, (cX, cY), 1, (255, 255, 255), -1)
            #cv2.putText(image, "center", (cX - 20, cY - 20),ArithmeticError
        	#	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            print("Kontur merkez x:",cX , "Kontur merkez y:", cY)
            
                 
    bolge_1, bolge_2 , bolge_3 , bolge_4 = [],[],[],[]      
    i =0
    l = len(cnt)
    while i < l:
  
        cnt_xy = cnt[i]   # x, y noktaları verir
        cnt_x, cnt_y, _, _ = cv2.boundingRect(cnt_xy)
        #print("cntxy : ",cnt_xy)
        if( cnt_x >= cX and cnt_y >= cY):
            #xy  = (cnt_x , cnt_y )
            xy  = np.array([cnt_x , cnt_y])
            bolge_1.append(xy)
            
            #cv2.circle(image, (cnt_x , cnt_y ), 1, (0, 255, 0), -1)
        i+=1
    i=0
    while i < l:
  
        cnt_xy = cnt[i]   # x, y noktaları verir
        cnt_x, cnt_y, _, _ = cv2.boundingRect(cnt_xy)
            
        if( cnt_x < cX and cnt_y >= cY ):
            
            xy  = np.array([cnt_x , cnt_y])
            bolge_2.append(xy)
            
            #cv2.circle(image, (cnt_x , cnt_y ), 1, (255, 255, 255), -1)
        i+=1
    i=0
    while i < l:
  
        cnt_xy = cnt[i]   # x, y noktaları verir
        cnt_x, cnt_y, _, _ = cv2.boundingRect(cnt_xy)
        
        if( cnt_x > cX and cnt_y <= cY ):
            
            xy  = np.array([cnt_x , cnt_y])
            bolge_3.append(xy)
            
            #cv2.circle(image, (cnt_x , cnt_y ), 1, (255, 255, 255), -1)
        i+=1
    i=0
    while i < l:
  
        cnt_xy = cnt[i]   # x, y noktaları verir
        cnt_x, cnt_y, _, _ = cv2.boundingRect(cnt_xy)
                  
        if( cnt_x < cX and cnt_y < cY):
           
            xy  = np.array([cnt_x , cnt_y])
            bolge_4.append(xy)
            
            #cv2.circle(image, (cnt_x , cnt_y ), 1, (0, 255, 0), -1)
        i+=1
        

    """    
    bolge1 = np.array(bolge_1)
    bolge2 = np.array(bolge_2)
    bolge3 = np.array(bolge_3)
    bolge4 = np.array(bolge_4)"""
    

    #cv2.circle(image, (bolge_1[0] ), 1, (0, 255, 0), -1)
    
    # bölge 1'i contour alanına dönüştürme
    l1= len(bolge_1)
    bolge_son = bolge_1 [l1-1]
    #cnt_x, cnt_y, _, _ = cv2.boundingRect(bolge_1 [ l1-1 ])
    print("bölge1 son nokta: ",bolge_son[0],bolge_son[1])

        
    x_çizgi = (bolge_son[0] - cX  )
    y_çizgi = (bolge_son[1] - cY  )
    print("bölge1 son nokta:",x_çizgi , y_çizgi)
    i = 0
    while i < x_çizgi:
        
        cnt_x = bolge_son[0]- 1
        cnt_y = bolge_son[1]
        xy  = (cnt_x , cnt_y )
        bolge_1.append(xy)
        
        l1= len(bolge_1)
        bolge_son = bolge_1 [l1-1]
        i+=1
        
    bolge_ilk = bolge_1 [0]
    x_çizgi = (bolge_ilk[0] - cX  )
    y_çizgi = (bolge_ilk[1] - cY  )
    print("bölge1 ilk nokta:",x_çizgi , y_çizgi)
    i = 0
    while i < y_çizgi-1:
        
        cnt_x = bolge_son[0]
        cnt_y = bolge_son[1]+1
        xy  = (cnt_x , cnt_y )
        bolge_1.append(xy)
        
        l1= len(bolge_1)
        bolge_son = bolge_1 [l1-1]
        i+=1
    # bölge 2'yi contour alanına dönüştürme
    l1= len(bolge_2)
    bolge_son = bolge_2 [l1-1]
    #cnt_x, cnt_y, _, _ = cv2.boundingRect(bolge_1 [ l1-1 ])
    print("bölge2 son nokta: ",bolge_son[0],bolge_son[1])

        
    x_çizgi = ( cX - bolge_son[0]   )
    y_çizgi = (bolge_son[1] - cY  )
    print("bölge2 son nokta: ",x_çizgi , y_çizgi)
    i = 0
    while i < y_çizgi:
        
        cnt_x = bolge_son[0]
        cnt_y = bolge_son[1]-1
        xy  = (cnt_x , cnt_y )
        bolge_2.append(xy)
        
        l1= len(bolge_2)
        bolge_son = bolge_2 [l1-1]
        i+=1
        
    bolge_ilk = bolge_2 [0]
    x_çizgi = (cX - bolge_ilk[0]  )
    y_çizgi = (bolge_ilk[1] - cY  )
    print("bölge2 ilk kordinat: ",bolge_ilk[0],bolge_ilk[1])
    print("bölge2 ilk nokta:",x_çizgi , y_çizgi)
    i = 0       
    while i < x_çizgi-1:
        
        cnt_x = bolge_son[0]- 1
        cnt_y = bolge_son[1]
        xy  = (cnt_x , cnt_y )
        bolge_2.append(xy)
        
        l1= len(bolge_2)
        bolge_son = bolge_2 [l1-1]
        i+=1
        
    # bölge 3'ü contour alanına dönüştürme
    l1= len(bolge_3)
    bolge_son = bolge_3 [l1-1]
    #cnt_x, cnt_y, _, _ = cv2.boundingRect(bolge_1 [ l1-1 ])
    print("bölge3 son nokta: ",bolge_son[0],bolge_son[1])

        
    x_çizgi = ( bolge_son[0] - cX )
    y_çizgi = ( cY - bolge_son[1] )
    print("bölge3 son nokta: ",x_çizgi , y_çizgi)
    i = 0
    while i < y_çizgi:
        
        cnt_x = bolge_son[0]
        cnt_y = bolge_son[1]+1
        xy  = (cnt_x , cnt_y )
        bolge_3.append(xy)
        
        l1= len(bolge_3)
        bolge_son = bolge_3 [l1-1]
        i+=1
        
    bolge_ilk = bolge_3 [0]
    x_çizgi = ( bolge_ilk[0] - cX )
    y_çizgi = ( cY - bolge_ilk[1]  )
    print("bölge3 ilk kordinat: ",bolge_ilk[0],bolge_ilk[1])
    print("bölge3 ilk nokta:",x_çizgi , y_çizgi)
    i = 0       
    while i < x_çizgi-1:
        
        cnt_x = bolge_son[0]+ 1
        cnt_y = bolge_son[1]
        xy  = (cnt_x , cnt_y )
        bolge_3.append(xy)
        
        l1= len(bolge_3)
        bolge_son = bolge_3 [l1-1]
        i+=1
        
    
        
    



    
    
    
    
    
    bolge1 = np.array(bolge_1)
    bolge2 = np.array(bolge_2)
    bolge3 = np.array(bolge_3)
    
    print("Bölge1 contour alanı:", cv2.contourArea(bolge1))
    print("Bölge2 contour alanı:", cv2.contourArea(bolge2))
    print("Bölge3 contour alanı:", cv2.contourArea(bolge3))
    print("en büyük contour alanı:", cv2.contourArea(cnt))  # en büyük contour alanı    
    
    #hull=cv2.convexHull(cnt)#dısbukey
    #cv2.drawContours (image,[hull], -1,(0,0,0), 7)

    
    stencil = np.zeros(image.shape).astype(image.dtype)
    cv2.fillPoly(stencil,  [cnt] , color=(255,255,255))
        
    #imshow('stencil', stencil) 
    stencil = cv2.cvtColor(stencil, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0 , 0 , 1, 0, 0],
              [0 , 1 , 1, 1, 0],
              [1 , 1 , 1, 1, 1],
              [0 , 1 , 1, 1, 0], 
              [0 , 0 , 1, 0, 0]], np.uint8)    
    closing = cv2.morphologyEx(stencil, cv2.MORPH_CLOSE, kernel, 2)
    cv2.imshow('closing', closing) 
    
    #np.where(closing>0, 1, 0)
    #   np.where(closing>0, 255, 0)
    #ben_indeks = np.where(closing>0 , closing, 0)
    ben_indeks = np.argwhere(closing >0)
    
    temp = np.zeros(image.shape).astype(image.dtype)
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    temp[ben_indeks[:,0], ben_indeks[:,1]]=255
    cv2.imshow('temp', temp) 
    #sub 2 index/ indeks 2 sub
    
    # i deki x i x+1 den x den  çıkar. farklar dizisin kareleri toplamı dx+dy nin kökü iki nokta için atlamaların genliği 
    #atlamaların ortalaması alınabilir. 
    
    
    # dx, dy türev alıp 2nci türeve döndür 4 tane ddx ddy türev genlikleri
    #
    # ASİMETRİ
    # SINIR
    # RENK 
   
    #ben = output
    #h=plt.hist(ben[:,:,0].ravel(),256,[0,256])  
    #plt.show()
    
    
    
    
    
    # ÇAP
    cap=radius
    print("Daire çapı:", ((cap*36)/100) ,'mm')
    
    #k = cv2.isContourConvex( cnt )  # konturun dış bukey olup olmadığını dönüyor
    #print(k)
    
    
    cv2.drawContours (image, cnt , -1, (0,255,0), 1)
    #cv2.drawContours(image, [max(contours, key = cv2.contourArea)], -1, (0,255,0) , thickness=-1)
    # drawcontour ile içini doldurma
    			 
		
    
    
    			
    
    # Showing the final image. 
    #cv2.imshow('image2', image) 
    
    
    
    
    
   
    #print( ben_indeks.shape())
    
    
    """
    h1=plt.hist(image[:,:,0].ravel(),256,[0,256])
    plt.show()
    h2=plt.hist(image[:,:,1].ravel(),256,[0,256])
    plt.show()
    h3=plt.hist(image[:,:,2].ravel(),256,[0,256])
    plt.show()
    thres=150
    m1 = image[:,:,0] < thres
    m2 = image[:,:,1] < thres
    m3 = image[:,:,2] < 200
    m =m1 & m2 & m3
    plt.imshow(m,cmap='gray')
    plt.show()                 """

    cv2.imshow("images", imageb)
    cv2.waitKey(0)