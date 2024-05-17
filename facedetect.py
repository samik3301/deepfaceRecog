import cv2
import dlib
import math

BLINK_RATIO_THRESHOLD = 5.7

# Calculating the blink ratio

def midpoint(point1 ,point2):
    return (point1.x + point2.x)/2,(point1.y + point2.y)/2

def euclidean_distance(point1 , point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_blink_ratio(eye_points, facial_landmarks):
    
    #loading all the required points
    corner_left  = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    corner_right = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    
    center_top    = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    #calculating distance
    horizontal_length = euclidean_distance(corner_left,corner_right)
    vertical_length = euclidean_distance(center_top,center_bottom)

    ratio = horizontal_length / vertical_length

    return ratio


# First we set the face classifier as Haar cascade - from opencv face detection

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# using dlib to extract facial features
detector = dlib.get_frontal_face_detector()

#get the eyes
predictor = dlib.shape_predictor("/Users/samik/Desktop/Programming/deepfaceRecog/shape_predictor_68_face_landmarks.dat")
#these landmarks are based on the image above 
left_eye_landmarks  = [36, 37, 38, 39, 40, 41]
right_eye_landmarks = [42, 43, 44, 45, 46, 47]

#Reading from live video stream 
video_capture = cv2.VideoCapture(0)


#Define a function to make bounding boxes around detected faces in the video stream

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(50, 50))
    for i,(x, y, w, h) in enumerate(faces): #for storing multiple detected faces in the stream
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 0, 0), 4) #draws a black rectangle
        # face = gray_image[y:y+h, x:x+w] #crops the detected face with its coordinates and saves it 
        # filename = f'/Users/samik/Desktop/Programming/deepfaceRecog/local_face_detect/detected_face_{i}.jpg'
        # cv2.imwrite(filename,face) #writes the cropped image into a local database
    return faces

def save_facial_snip(vid,save_face):
    if save_face:
        gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(50, 50))
        for i,(x, y, w, h) in enumerate(faces): #for storing multiple detected faces in the stream
            #cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 0, 0), 4) #draws a black rectangle
            face = gray_image[y-50:y+h+50, x-50:x+w+50] #crops the detected face with its coordinates and saves it 
            filename = f'/Users/samik/Desktop/Programming/deepfaceRecog/local_face_detect/detected_face_{i}.jpg'
            cv2.imwrite(filename,face) #writes the cropped image into a local database
    

live_person = False #global variable


save_face=False # Don't save face until a blink is detected- ie a person is live

# Run time loop to get video frames
while True:
    
    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully

    faces2 = detect_bounding_box(video_frame)  # apply the draw bounding box function to the video frame
    save_facial_snip(video_frame,save_face)

    #detecting faces in the frame 
    faces,_,_ = detector.run(image = video_frame, upsample_num_times = 0, adjust_threshold = 0.0)

    # Detecting Eyes using landmarks in dlib-----
    for face in faces:
    
        landmarks = predictor(video_frame, face)

        #Calculating blink ratio for eye-----
        left_eye_ratio  = get_blink_ratio(left_eye_landmarks, landmarks)
        right_eye_ratio = get_blink_ratio(right_eye_landmarks, landmarks)
        blink_ratio     = (left_eye_ratio + right_eye_ratio) / 2

        liveness_flag= False
        
        if blink_ratio > BLINK_RATIO_THRESHOLD:
            #Blink detected! 
            gray_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
            for i,(x, y, w, h) in enumerate(faces2): #for storing multiple detected faces in the stream
                cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 4) #draws a green rectangle
            
            save_face = True #save only the face after the person blinks - even if it detects false positives it won't save those

            liveness_flag = True
            live_person=True
            #cv2.putText(video_frame,"BLINKING",(10,50), cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
        if live_person==False:
            cv2.putText(video_frame,"Blinking not Detected",(10,110), cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),2,cv2.LINE_AA)

    if live_person==True:
        cv2.putText(video_frame,"Live person in frame",(10,110), cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),2,cv2.LINE_AA)


    cv2.imshow("Face Detection", video_frame)  

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()