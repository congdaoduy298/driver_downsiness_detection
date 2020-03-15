# driver_downsiness_detection

This project include : 

 + get_training_data.py: create folder close eye images and open eye images before training model.
 
 + training_model.ipynb: after create data, using that data to training weights of CNN. Save the weights 
 
 and architectures of model to : model.h5 file 
 
 + downsiness_detection.py: detect faces and detect eyes in the biggest face (using pretrained Haar Cascade). 
 
 Put eyes image into the model and get the predicted eye state. Create a function to calculate score dangerous.
 
 If it is more than 10 mark then alert driver immediately.
 
 
 
