import cv2
import numpy as np
import os

#img is a numpy array it has col and each col is a array with height and width each element is pixel value
img_path="opencv\pp.JPG"
if not os.path.exists(img_path):
    print("File not found:", "/pp.JPG")
    exit()
img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
#cv2.IMREAD_COLOR Just displays normal picture
#cv2.IMREAD_GRAYSCALE makes picture gray

# print(img.size)#number of pixels
# print(img.shape)#gives number of pixel rows and colums


cv2.imshow("Profile pic",img)
cv2.waitKey(0)#window closes when u hit a key
cv2.destroyAllWindows()#once window close destroy all windows

#storing gray image to the file 
# cv2.imwrite("graypic.JPG",img)

# Resize image
"""
img2=cv2.resize(img,(200,200))
img2=cv2.resize(img,(0,0),fx=2,fy=1)#doubling x dimension and keeping y constant
cv2.imshow("Profile pic",img2)
cv2.waitKey(0)
cv2.destroyAllWindows()"""


#cropping image
"""
height,width=img.shape[0],img.shape[1]#getting height and width
img=img[int(height/2):-int(height/3),50:-50]
#before comma is the height from where : to top height/2 is cropped till 
#minus means neeche seh
#so height/3 cropped from neeche 
#if height/2:height only top wld crop
#width is shown as pixel so basically front and back seh 50 pixel crop
cv2.imshow("Profile pic",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

#Rotate Image
"""

img=cv2.rotate(img,cv2.ROTATE_180)#rotating upside down

cv2.imshow("Profile pic",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

#Drawing shapes on image

#Border
"""
img=cv2.copyMakeBorder(img,20,20,20,20,borderType=cv2.BORDER_CONSTANT,value=(100,0,0))#20px border on TLBR and value is basically gbr code
cv2.imshow("Profile pic",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

#line
"""
img=cv2.line(img,(50,50),(500,50),color=(0,200,0),thickness=20)
#50,50 is start coordinate 500 50 is end coordinate
cv2.imshow("Profile pic",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

#arrow
"""
img=cv2.arrowedLine(img,(50,50),(500,50),color=(0,200,0),thickness=20)
#50,50 is start coordinate 500 50 is end coordinate
cv2.imshow("Profile pic",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

#circle
"""
img=cv2.circle(img,(500,500),color=(0,200,0),radius=20,thickness=20)
#500 500 is the centre
# if thickness is negative it will give filled circle
cv2.imshow("Profile pic",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

#rectangle
"""
img=cv2.rectangle(img,pt1=(100,100),pt2=(500,500),color=(150,150,150),thickness=20)
#top left and bottom right corner
cv2.imshow("Profile pic",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
#text
"""
img=cv2.putText(img,"Hello Pratham",org=(0,400),thickness=2,fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=1,color=(0,0,0))
#400 400 is bottom left corner
cv2.imshow("Profile pic",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


