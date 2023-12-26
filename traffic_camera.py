import numpy as np
import cv2
import pickle 
import tensorflow as tf

threshold = 0.80     #PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX

#setup the video camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 180)

my_model = tf.keras.models.load_model('traffic_model.h5')
my_model.load_weights('weights_model.h5')

def preprocessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

def getCalssName(classNo):
    if classNo == 0: return 'Speed limit (20km/h)'
    elif classNo == 1:return 'Speed limit (30km/h)'
    elif classNo == 2:return 'Speed limit (50km/h)'
    elif classNo == 3:return 'Speed limit (60km/h)'
    elif classNo == 4:return 'Speed limit (70km/h)'
    elif classNo == 5:return 'Speed limit (80km/h)'
    elif classNo == 6:return 'End of speed limit (80km/h)'
    elif classNo == 7:return 'Speed limit (100km/h)'
    elif classNo == 8:return 'Speed limit (120km/h)'
    elif classNo == 9:return 'No passing'
    elif classNo == 10:return 'No passing veh over 3.5 tons'
    elif classNo == 11:return 'Right-of-way at intersection'
    elif classNo == 12:return 'Priority road'
    elif classNo == 13:return 'Yield'
    elif classNo == 14:return 'Stop'
    elif classNo == 15:return 'No vehicles'
    elif classNo == 16:return 'Veh > 3.5 tons prohibited'
    elif classNo == 17:return 'No entry'
    elif classNo == 18:return 'General caution'
    elif classNo == 19:return 'Dangerous curve left'
    elif classNo == 20:return 'Dangerous curve right'
    elif classNo == 21:return 'Double curve'
    elif classNo == 22: return 'Bumpy road'
    elif classNo == 23: return 'Slippery road'
    elif classNo == 24: return 'Road narrows on the right'
    elif classNo == 25: return 'Road work'
    elif classNo == 26: return 'Traffic signals'
    elif classNo == 27: return 'Pedestrians'
    elif classNo == 28: return 'Children crossing'
    elif classNo == 29: return 'Bicycles crossing'
    elif classNo == 30: return 'Beware of ice/snow'
    elif classNo == 31: return 'Wild animals crossing'
    elif classNo == 32: return 'End speed + passing limits'
    elif classNo == 33: return 'Turn right ahead'
    elif classNo == 34: return 'Turn left ahead'
    elif classNo == 35: return 'Ahead only'
    elif classNo == 36: return 'Go straight or right'
    elif classNo == 37: return 'Go straight or left'
    elif classNo == 38: return 'Keep right'
    elif classNo == 39: return 'Keep left'
    elif classNo == 40: return 'Roundabout mandatory'
    elif classNo == 41: return 'End of no passing'
    elif classNo == 42: return 'End no passing veh > 3.5'
while True:
    # Read image
    success, imgOrignal = cap.read()

    # Process image
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, dsize=(32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = np.array(img).reshape(1,32,32,1)
    cv2.putText(imgOrignal, "TRAFFIC SIGN: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    
    # PREDICT IMAGE
    predictions = my_model.predict(img) #
    classIndex = np.argmax(predictions,axis=1)
    probabilityValue = np.amax(predictions)

    if probabilityValue > threshold:
        #print(getCalssName(classIndex))
        cv2.putText(imgOrignal, str(classIndex)+" "+str(getCalssName(classIndex)), 
                    (185, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue*100, 2))+"%", 
                    (185, 75), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Result", imgOrignal)
        
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break