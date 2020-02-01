import tkinter as tk
import cv2
import os
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

def predictiontime():
    test_img=cv2.imread("test-data/test.jpg")
    predicted_img=predict(test_img)
    cv2.imshow("test",cv2.resize(predicted_img,(400,500)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def predictall():
    testsubs = os.listdir("test-data")
    for imagename in testsubs:
        if imagename.startswith("."):
                continue;
        imagepath = "test-data/" + imagename
        test_img=cv2.imread(imagepath)
        predicted_img=predict(test_img)
        cv2.imshow(imagename,cv2.resize(predicted_img,(400,500)))
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
        
def videoCap():
    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(0)
    while True:   
        check, frame = webcam.read()
        print(check)
        print(frame)
        frame=cv2.flip(frame,1)
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'): 
            #os.chdir(r'C:\Users\admin\Desktop\facerec\test-data')
            #fname='webcamtest'+str(random.randint(100,200))+'.jpg'
            #cv2.imwrite(filename=fname, img=frame)
            webcam.release()
            cv2.imshow('Captured image',frame)
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
            #os.chdir(r'C:\Users\admin\Desktop\facerec')
            #print("Image saved!")
            #image=cv2.imread("test-data/"+fname)
            img=predict(frame)
            cv2.imshow('Captured image prediction',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break
        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break

#------------Main--------------#
subjects = ["Aditya","Amber Heard","Bill Gates","Jason Momoa","Paul Rudd","Scarlett Johansson"]
face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer.read('eigen.cv2')
m=tk.Tk()
m.title("Facial detection")
w1=tk.Button(master=m, command=predictiontime,text='Detect showcase main')
w2=tk.Button(master=m,command=predictall,text='Detect showcase all')
w3=tk.Button(master=m,command=videoCap,text='Webcam showcase')
w1.pack()
w2.pack()
w3.pack()
m.mainloop()