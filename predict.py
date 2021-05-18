# -*- coding: utf-8 -*-
"""
Created on Tue May 18 23:17:56 2021

@author: Saif
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


classes_list = sorted(['butterfly',  'cougar_face', 'elephant'])
# Enter your class names in this list
global label_names

# Must be same as Annotations list we used to choose the data
label_names = sorted(classes_list)

model = load_model('caltech.h5')

"""### **Preprocessing Funciton**
This function will preprocess new images in the same way we did initially before training.
"""

# This function will preprocess images.
def preprocess(img, image_size = 300):
  
    image = cv2.resize(img, (image_size, image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float") / 255.0 

    # Expand dimensions as predict expect image in batches
    image = np.expand_dims(image, axis=0) 
    return image

"""### **Post Processing Function**
After the prediction, you need to do some postprocessing in order to extract the class label and the real boudning box coordinates.
"""

def postprocess(image, results):

    # Split the results into class probabilities and box coordinates
    bounding_box, class_probs = results

    # First let's get the class label

    # The index of class with the highest confidence is our target class
    class_index = np.argmax(class_probs)
  
    # Use this index to get the class name.
    class_label = label_names[class_index]

    # Now you can extract the bounding box too.

    # Get the height and width of the actual image
    h, w = image.shape[:2]

    # Extract the Coordinates
    x1, y1, x2, y2 = bounding_box[0]

    # Convert the coordinates from relative (i.e. 0-1) to actual values
    x1 = int(w * x1)
    x2 = int(w * x2)
    y1 = int(h * y1)
    y2 = int(h * y2)

    # return the lable and coordinates
    return class_label, (x1,y1,x2,y2),class_probs


"""### **Predict Function**
Here we will create a function that that will use the model to predict on new images.

"""

def predict(image, returnimage = False,  scale = 0.9):
  
  # Before we can make a prediction we need to preprocess the image.
  processed_image = preprocess(image)

  # Now we can use our model for prediction
  results = model.predict(processed_image)

  # Now we need to postprocess these results.
  # After postprocessing, we can easily use our results
  label, (x1, y1, x2, y2), confidence = postprocess(image, results)

  # Now annotate the image
  cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 100), 2)
  cv2.putText(
      image, 
      '{}'.format(label, confidence), 
      (x1, y2 + int(35 * scale)), 
      cv2.FONT_HERSHEY_COMPLEX, scale,
      (200, 55, 100),
      2
      )

  # Show the Image with matplotlib
  plt.figure(figsize=(10,10))
  plt.imshow(image[:,:,::-1])
  
image = cv2.imread('1.jpg')
predict(image)