import cv2
import tkinter as tk
from PIL import ImageTk, Image

#------------------------------------FUNCTIONS-----------------------------------------------------#

def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')
    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);   
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None   
    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0] 
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def predict(test_img):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    image2=cv2.resize(img,(400,500))
    face, rect = detect_face(image2)
    #predict the image using our face recognizer 
    if face is not None:
        label, confidence = face_recognizer.predict(cv2.resize(face,(100,200)))
        #get name of respective label returned by face recognizer
        label_text = subjects[label]
        
        #draw a rectangle around face detected
        draw_rectangle(image2, rect)
        #draw name of predicted person
        draw_text(image2, label_text, rect[0], rect[1]-5)
        
        return image2
    else:
        draw_text(image2,"No face detected",6,25)
        return image2
        
#-------------------MAIN-------------------#

m=tk.Tk()
subjects = ["Aditya","Amber Heard","Bill Gates","Jason Momoa","Paul Rudd","Scarlett Johansson"]
face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer.read('eigen.cv2')

def helper(label_text):
    if(label_text=="Aditya"):
        m.destroy()
        s=tk.Tk()
        s.title("Unlocked!")
        img=ImageTk.PhotoImage(Image.open("unlock.png"))
        panel=tk.Label(s,image=img)
        panel.pack(side="bottom",fill="both",expand="yes")
        s.mainloop()
    else:
        w.pack_forget()
        locked=tk.Message(master=m,text="Locked!",width=100)
        locked.pack()
def Unlock():
    label_text="None"
    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(0)
    while True:   
        check, frame = webcam.read()
        print(check)
        print(frame)
        frame=cv2.flip(frame,1)
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('u'): 
            webcam.release()
            cv2.imshow('Captured image',frame)
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
            image=cv2.resize(frame,(400,500))
            face, rect = detect_face(image)
            if face is not None:
                label, confidence = face_recognizer.predict(cv2.resize(face,(100,200)))
                label_text = subjects[label]
            break
        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
    helper(label_text)
def unlockFromFile():
    frame=cv2.imread("test-data/unlock.jpg")
    image=cv2.resize(frame,(400,500))
    face, rect = detect_face(image)
    if face is not None:
        label, confidence = face_recognizer.predict(cv2.resize(face,(100,200)))
    label_text = subjects[label]
    helper(label_text)


m.title("Face unlock")
img=ImageTk.PhotoImage(Image.open("lock.png"))
panel=tk.Label(m,image=img)
panel.pack(side="bottom",fill="both",expand="yes")
w=tk.Button(master=m,command=Unlock,text="Unlock")
wx=tk.Button(master=m,command=unlockFromFile,text="Unlock from file")
w.pack()
wx.pack()
m.mainloop()