import cv2
import cv2.data
import numpy 
#opening web cam
"""
stream=cv2.VideoCapture(0)#0 means webcam
if not stream.isOpened():
    print("No Cam")
    exit()

while(True):
    ret,frame=stream.read()
    if not ret:
        print("No more cam")
        break
    cv2.imshow("Webcam",frame)
    #if q is pressed exit the while loop
    if cv2.waitKey(1)==ord('q'):
        break

stream.release()
cv2.destroyAllWindows()
"""

##storing the captured video
"""
stream=cv2.VideoCapture(0)#0 means webcam
if not stream.isOpened():
    print("No Cam")
    exit()
#getting the video ka properties
fps=stream.get(cv2.CAP_PROP_FPS)
height=int(stream.get(4))
width=int(stream.get(3))
#setting up to store the stream
output=cv2.VideoWriter("./stream.mp4",cv2.VideoWriter_fourcc('m','p','4','v'),fps=fps,frameSize=(width,height))

while(True):
    ret,frame=stream.read()
    if not ret:
        print("No more cam")
        break
    frame=cv2.resize(frame,(width,height))
    #all frames are stored
    output.write(frame)
    cv2.imshow("Webcam",frame)
    #if q is pressed exit the while loop
    if cv2.waitKey(1)==ord('q'):
        break

stream.release()
cv2.destroyAllWindows()
"""
#corner detection
"""
img=cv2.imread("./shapes.jpeg")
img=cv2.resize(img,(600,600))
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#just gray better for corner detection

##method1 not so good
"""
"""
##there is a small box which slides when a line enters at an angle it predicts where
#it exits if prediction is right no corner is it exits at another angle then line must have bent 
#in the box so corner exists
#based on the size of the difference the quality of the corner

corners=cv2.goodFeaturesToTrack(gray_img,maxCorners=50,qualityLevel=0.12,minDistance=2)
#corners is a list of x y coordinates
corners = corners.astype(int)
#converting to int format
for c in corners:
    x,y=c.ravel()
    img=cv2.circle(gray_img,center=(x,y),radius=2,color=(0,0,255),thickness=-1)
cv2.imshow("shapes",gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

#method 2
"""
corners=cv2.goodFeaturesToTrack(gray_img,maxCorners=50,qualityLevel=0.01,minDistance=2,useHarrisDetector=True,k=0.1)
corners = corners.astype(int)
#converting to int format
for c in corners:
    x,y=c.ravel()
    img=cv2.circle(gray_img,center=(x,y),radius=2,color=(0,0,255),thickness=-1)
cv2.imshow("shapes",gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
""""""
#face detection(detecting features)

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
smile_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_smile.xml")
eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye.xml")
def detect_features(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,minNeighbors=5)#5 is min number of neighbours#1.3 is scale
    #now draw rectangle for faces
    for (x,y,w,h) in faces:
        frame=cv2.rectangle(frame,(x,y),(x+w,y+h),color=(0,255,0),thickness=5)
        face=frame[y:y+h,x:x+w]
        gray_face=gray[y:y+h,x:x+w]
        smiles=smile_cascade.detectMultiScale(gray_face,2.5,minNeighbors=9)
        for (xp,yp,wp,hp) in smiles:
            face=cv2.rectangle(face,(xp,yp),(xp+wp,yp+hp),color=(0,0,255),thickness=5)
        eyes=eye_cascade.detectMultiScale(gray_face,2.5,minNeighbors=5)#inc minnbrs to make more accuracy or strict
        for (xp,yp,wp,hp) in eyes:
            face=cv2.rectangle(face,(xp,yp),(xp+wp,yp+hp),color=(255,0,0),thickness=5)  
    return frame
stream=cv2.VideoCapture(0)
if not stream.isOpened():
    print("No Cam")
    exit()

while(True):
    ret,frame=stream.read()
    if not ret:
        print("No more cam")
        break
    frame=detect_features(frame)#only after detecting display frame
    cv2.imshow("Webcam",frame)
    if cv2.waitKey(1)==ord('q'):
        break

stream.release()
cv2.destroyAllWindows()