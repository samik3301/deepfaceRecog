from deepface import DeepFace
DeepFace.stream(db_path = "/Users/samik/Desktop/Programming/deepfaceRecog/db",model_name="VGG-Face",detector_backend="opencv",enable_face_analysis=False,time_threshold=2,frame_threshold=1)
'''
RetinaFace and MTCNN seem to overperform in detection and alignment stages but they are much slower. 
If the speed of your pipeline is more important, then you should use opencv or ssd. On the other hand, 
if you consider the accuracy, then you should use retinaface or mtcnn.

In our use case, I am prioritizing the speed in the speed-accuracy tradeoff and going with opencv face detector backend.

For choosing the model, we can either go with Facenet512 which has a LWF Score of 99.65% or we can go with traditional VGG-Face 
which has a LWF score of 98.78% and YTF score of 97.40%.

'''