# driver_downsiness_detection

This project include : 

 + get_training_data.py: create folder close eye images and open eye images before training model.
 
 + training_model.ipynb: after create data, using that data to training weights of CNN. Save the weights 
 
    and architectures of model to file model.h5.
 
 + downsiness_detection.py: detect faces and detect eyes in the biggest face (using pretrained Haar Cascade). 
 
   Put eyes image into the model and get the predicted eye state. Create a function to calculate score dangerous.
 
   If it is more than 10 mark then alert driver immediately.
 
 
 # Reference
 
 Intermediate Python Project – Driver Drowsiness Detection System with OpenCV & Keras
 https://data-flair.training/blogs/python-project-driver-drowsiness-detection-system/
 
 Haar Cascade Object Detection Face & Eye OpenCV Python Tutorial
 https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/
