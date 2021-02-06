import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from HsvDeger import karart
import csv
import glob


#özellik dizisi olusturma 
features=[0,0,0,0,0,0,0]

cv_img = []
image_namelist = []
for img in glob.glob("./foto2/*.png"):
    image_namelist.append(img)
    n = cv2.imread(img)
    cv_img.append(n)


for s,image_as in enumerate(cv_img):
    print("\n ---- Foto: ",s,"-------\n")
    img_clone = image_as.copy()

    #--------Görselin hsv değerlerini düzenlemek için karart fonksiyonu
    
    image = karart(image_as)
    
#----------------------Lezyon Bölütleme  
    
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Sınırlardan NumPy dizileri oluşturma
    lower = np.array([0, 100, 19], dtype = "uint8")
    upper = np.array([17, 200, 120], dtype = "uint8")
     
    # Belirlenen sınırlar içindeki renklerin bulunması ve maskenin uygulanması
    mask  = cv2.inRange(converted, lower, upper)
    plt.imshow(mask)
    plt.show()
    
    output = cv2.bitwise_and(image, image, mask = mask )
    #plt.imshow(output)
    #plt.show()
    
    
    #------------------Maske üzerinden konturları bulma
    contours, _ = cv2.findContours(mask , cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    
    
    
    # ----------------Lezyon etrafına çember çizdirme
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    print("daire merkez",center)
    radius = int(radius)
    imageb = cv2.circle(image,center,radius,(0,255,0),2)
    
    
    #Lezyon etrafındaki konturun çizdirilmesi / Lezyonun kenarlarını çizme
    #cv2.drawContours (image, cnt , -1, (0,255,0), 1)
    cv2.drawContours (image, contours , -1, (0,255,0), 1)
    
    
    # ---------------En büyük konturun merkez koorinatlarını bulma
    for c in contours:        
        if cv2.contourArea(c) == (cv2.contourArea(cnt)) :
            
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
                     
            
        	# Konturun merkezini görselin üzerine çizme
            cv2.circle(image, (cX, cY), 1, (255, 255, 255), -1)
            
            print("Kontur merkez x:",cX , "Kontur merkez y:", cY)
          
#----  Lezyon Sınır Noktalarının Merkeze Olan Uzaklık Ortalamasından yüksek olanları belirleme            
    merk_uzunlist = []        
    i,l = 0,0
    l = len(cnt)
    merk_toplam = 0
    while i < l:
        i_xy = cnt[i]   # x, y noktaları verir
        i_x, i_y, _, _ = cv2.boundingRect(i_xy)
        
        
        merk_uzunluk = np.sqrt((abs(i_x-cX))*(abs(i_x-cX))+(abs(i_y-cY))*(abs(i_y-cY)))
        merk_toplam = merk_toplam + merk_uzunluk
        merk_uzunlist.append(merk_uzunluk)
        
        i+=1
        
        
    merk_ort = merk_toplam / len(merk_uzunlist)
    print("Lezyon Sınır Noktalarının Merkeze Olan Uzaklık Ortalaması:",merk_ort)
    #----  Lezyon Sınır Noktalarının Merkeze Olan Uzaklık Ortalamasından yüksek olanları belirleme
    farklı_uzunluk = 0
    esik = abs( merk_ort - (merk_ort*(70/100)))
    for d in merk_uzunlist:
        if( ( d > ( merk_ort + esik ) ) or ( d < ( merk_ort - esik ) ) ):
            farklı_uzunluk +=1
    
    print("ortalamadan yüksek uzunluk sayısı:", farklı_uzunluk)
    
    features[0]=farklı_uzunluk
       
    
    # ------En büyük contour alanı
    print("en büyük contour alanı:", cv2.contourArea(cnt))      
    
    #-------Lezyon konturunun kenarlarını kapama işlemi ile yumuşatma
    
    stencil = np.zeros(image.shape).astype(image.dtype)
    cv2.fillPoly(stencil,  [cnt] , color=(255,255,255))
        
     
    stencil = cv2.cvtColor(stencil, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0 , 0 , 1, 0, 0],
              [0 , 1 , 1, 1, 0],
              [1 , 1 , 1, 1, 1],
              [0 , 1 , 1, 1, 0], 
              [0 , 0 , 1, 0, 0]], np.uint8)    
    closing = cv2.morphologyEx(stencil, cv2.MORPH_CLOSE, kernel, 2)
    #cv2.imshow('closing', closing) 
        
    
    
    ben_indeks = np.argwhere(closing >0)
    
    temp = np.zeros(image.shape).astype(image.dtype)
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    temp[ben_indeks[:,0], ben_indeks[:,1]]=255
    #cv2.imwrite("temp.png", temp)   
    
    
    
    
# -----------------Sınır Düzensizliği
    
    #----Kontur noktalarının arasındaki genliği, 1.dereceden türev ile hesaplama
    genlik_list = []
    turev_konum = []
    i ,j,genliksum,l = 3,0,0,0
    l = len(cnt)
    
    while i < l:
        i_xy = cnt[i]   # x, y noktaları verir
        i_x, i_y, _, _ = cv2.boundingRect(i_xy)
        
        j_xy = cnt[j]   # x, y noktaları verir
        j_x, j_y, _, _ = cv2.boundingRect(j_xy)
        genlik = np.sqrt((abs(i_x-j_x))*(abs(i_x-j_x))+(abs(i_y-j_y))*(abs(i_y-j_y)))
        genliksum = genliksum + genlik
        genlik_list.append(genlik)
        #print("genlik",genlik)
        turev_konum.append([(i_x-j_x),(i_y-j_y)])
        i+=3
        j+=3
        
    
    genlikort = genliksum / len(genlik_list)
    print("genlik ortalaması 1(1.Türev):",genlikort)
    #----  1.dereceden türeve göre Genlik ortalaması eşikten yüksek olanları belirleme
    farklı_genlik = 0
    for b in genlik_list:
        if( (b-genlikort) > 0.5 ):
            farklı_genlik +=1
    
    print("farklı genlik sayısı(1.türev):", farklı_genlik)
    
    
    #---- 1.dereceden Türevi alınmış noktarın genliğini, 2.dereceden türev ile hesaplama
    
    genlik_list2 = []
    i ,j,genliksum2,genlik,l = 0,1,0,0,0
    l = len(turev_konum)
    turev_konum = np.array(turev_konum)
    
    while i < l-1:
        i_x = turev_konum[i,0] 
        # x, y noktaları verir
        i_y = turev_konum[i,1]
     
        
        j_x = turev_konum[j,0] 
        # x, y noktaları verir
        j_y = turev_konum[j,1]
        genlik = np.sqrt((abs(i_x-j_x))*(abs(i_x-j_x))+(abs(i_y-j_y))*(abs(i_y-j_y)))
        genliksum2 = genliksum2 + genlik
        genlik_list2.append(genlik)
        #print("genlik2: ",genlik)
        
        i+=1
        j+=1
        
    genlikort2 = genliksum2 / len(genlik_list2)
    print("genlik ortalaması2: ",genlikort2)
    
    #----  2.dereceden türeve göre Genlik ortalaması eşikten yüksek olanları belirleme
    farklı_genlik = 0
    for a in genlik_list2:
        if( (a-genlikort2) > 1 ):
            farklı_genlik +=1
    
    print("farklı 2. türev genlik sayısı:", farklı_genlik)
    features[1]=farklı_genlik   
        
    
    
#-------------------------------RENK 
    #------- Renkli fotoğrafta ROI
    
    i,j=0,0
    renklicnt = np.zeros(image.shape).astype(image.dtype)
    while i < 256:      
        j=0
        while j < 256:
            
            if( closing[i,j] == 255 ):
                renklicnt[i,j] = img_clone[i,j]
                
            j+=1
        i+=1
        
    #cv2.imshow("Renkli roi",renklicnt)
    renklicnt_clone = renklicnt.copy()
    
    
    
    #-----------Lezyondaki renklerin standart sapmasın hesaplanması
    
    renkliben = renklicnt[ben_indeks[:,0],ben_indeks[:,1],:]
    
    arr=np.asarray(renkliben)
    img_hsv = colors.rgb_to_hsv(arr[...,:3])
    
    lu1=img_hsv[...,0].flatten()
    plt.subplot(1,3,1)
    renkhist = plt.hist(lu1,bins=20, histtype='stepfilled', color='r', label='Hue')
    
    
    frekans_ort = np.mean(renkhist[0])
    
    renk_indeksler = np.argwhere(renkhist[0]>frekans_ort)
    
    renkler = renk_indeksler*0.025
    std_renk = np.std(renkler)
    features[2]=round(std_renk,3)
    print("std_renk:",std_renk)
    
    plt.title("Hue")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    
    
#----------------------- ÇAP
    
    cap=((radius*135)/1000)*2
    print("Daire çapı:", (cap) ,'mm')
    features[3]=cap
    
    
     
#--------------ASİMETRİ
    # Temp üzerinden tekrar sınırı bulup ROI'yi çıkarma
    thresh = cv2.threshold(temp, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        roi = temp[y:y+h, x:x+w]
        break
    
    cv2.imshow('ROI',roi)
    
    print("roi asıl",roi.shape)    
        
    height, width = roi.shape
    
    width_cutoff = width // 2
    
    
    
    # Dice similarity fonksiyonu
    def dice(pred, true, k = 255):
        intersection = np.sum(pred[true==k]) * 2.0
        dice = intersection / (np.sum(pred) + np.sum(true))
        return dice
    
    
    
    
    
    #ROI ortadan ikiye bölme ve Lezyonun sağ yarısı ile sol yarısının benzerliğini hesaplama
    #(Asimetrisine bakma)
    try:
        roi1 = roi[:, :(width_cutoff)] #left half
        roi2 = roi[:, (width_cutoff):] #right half
        
        roi2_mirror = cv2.flip(roi2, 1)
        #cv2.imshow('right mirror', roi2_mirror)
                
        dice_score = dice(roi1, roi2_mirror) 
        
    except IndexError:
        roi1 = roi[:, :(width_cutoff+1)] #left half
        roi2 = roi[:, (width_cutoff):] #right half
        
        roi2_mirror = cv2.flip(roi2, 1)
        #cv2.imshow('right mirror', roi2_mirror)
                
        dice_score = dice(roi1, roi2_mirror)
        
          
    print("roi1",roi1.shape)
    print("roi2",roi2.shape)
    features[4]=round(dice_score,3)
    print ("Dice Similarity: ", dice_score)
    
    # Save each half   
    #cv2.imwrite("left.png", roi1) 
    #cv2.imwrite("right.png", roi2) 
    
        
    
#----------------- Sadece Lezyon içindeki konturların bulunması
       
    
    renklicnt_clone = karart(renklicnt_clone)
    converted2 = cv2.cvtColor(renklicnt_clone, cv2.COLOR_BGR2HSV)
    # Sınırlardan NumPy dizileri oluşturma
    lower = np.array([0, 100, 19], dtype = "uint8")
    upper = np.array([17, 200, 120], dtype = "uint8")
     
    # Belirlenen sınırlar içindeki renklerin bulunması ve maskenin uygulanması
    mask2  = cv2.inRange(converted2, lower, upper)
    
    output2 = cv2.bitwise_and(renklicnt_clone, renklicnt_clone, mask = mask2 )

    
    
    #------------------Maske üzerinden konturları bulma
    contours2, _ = cv2.findContours(mask2 , cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    print("Lezyon içindeki Kontur sayisi:",len(contours2))
    features[5]=len(contours2)
    
    cv2.drawContours (renklicnt_clone, contours2 , -1, (0,255,0), 1)
    cv2.imshow("renklicnt_kontur", renklicnt_clone)
    
    
    
#------------- Hedef özelliğini cvs dosyasına yazma

    image_name = image_namelist[s]
    image_name = image_name[8:9]
    print("fotoğraf harfi: ",image_name)
    if (image_name == 'h'):
        features[6]= 1
        print("hasta!")
    elif (image_name == 'n'):
        features[6]= 0
        print("sağlıklı!")
    else:
        print("Görsel isimlendirmesi yanlış!")
          
    
    print(features)
    with open('lezyon.csv', mode='a',newline='') as yeni_dosya:
        yeni_yazici = csv.writer(yeni_dosya, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        yeni_yazici.writerow(features)
          
         
    cv2.imshow("images", image)
    

cv2.waitKey(0)
