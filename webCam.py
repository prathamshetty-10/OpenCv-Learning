import cv2

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