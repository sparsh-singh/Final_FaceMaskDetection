import cv2
import numpy as np
from tensorflow.keras.models import load_model
model=load_model("freshface.h5")#load the model
#model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

source=cv2.VideoCapture(0)

labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}
while(True):
    
    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(gray,1.3,5)  
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    if len(faces)==0:
            
        print("faces",faces)
        resized=cv2.resize(img,(150,150))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,150,150,3))
        result=model.predict_classes(reshaped)
        print("hai",result)
    
            #label=np.argmax(result,axis=1)[0]
        print(result[0][0])
        cv2.rectangle(img,(0,0),(200,100),color_dict[result[0][0]],-1)
        cv2.putText(img, labels_dict[result[0][0]], (50, 50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)    
            

    for (x,y,w,h) in faces:
  
        face_img=img[y:y+h,x:x+w]
                
        resized=cv2.resize(img,(150,150))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,150,150,3))
        result=model.predict_classes(reshaped)
        print("hai",result)

        #label=np.argmax(result,axis=1)[0]
        print(result[0][0])
     
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[result[0][0]],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[result[0][0]],-1)
        cv2.putText(img, labels_dict[result[0][0]], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()