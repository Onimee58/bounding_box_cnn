# -*- coding: utf-8 -*-
"""
Created on Tue May 18 22:27:19 2021

@author: Saif
"""

import cv2
import scipy.io
import os
import pandas as pd
import imageio
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import re
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Activation, Input, Dropout
from tensorflow.keras.optimizers import Adam, SGD


"""
The function will be used in the function below to go through the image 
and annotation directories and extract their information such as 
* Height
* Width
* Bounding box coordinates
* Class
"""

def extract_mat_contents(annot_directory, image_dir):
        
        # Create MAT Parser
        mat = scipy.io.loadmat(annot_directory)

        # Get the height and width for our image
        height, width = cv2.imread(image_dir).shape[:2]

        # Get the bounding box co-ordinates
        x1, y2, y1, x2 = tuple(map(tuple, mat['box_coord']))[0]

        # We Split the image Directory passed in the method and choose the index
        # Of the Folders name which is the same as it's class
        class_name = image_dir.split('/')[2]

        filename = '/'.join(image_dir.split('/')[-2:])

        # Return the extracted attributes
        return filename,  width, height, class_name, x1,y1,x2,y2
    
""" 
Then this information will be stored in a pandas dataframe so we can
use it with imgaug to perform augmentations.
"""

def mat_to_csv(annot_directory, image_directory, classes_folders):

  # List containing all our attributes regarding each image
  mat_list = []

  # We loop our each class and its labels one by one to preprocess and augment 
  for class_folder in classes_folders:

    # Set our images and annotations directory
    image_dir = os.path.join(image_directory, class_folder)
    annot_dir = os.path.join(annot_directory, class_folder) 

    # Get each file in the image and annotation directory
    mat_files = sorted(os.listdir(annot_dir))
    img_files = sorted(os.listdir(image_dir))

    # Loop over each of the image and its label
    for mat, image_file in zip(mat_files, img_files):
      
      # Full mat path
      mat_path = os.path.join(annot_dir, mat)

      # Full path Image
      img_path = os.path.join(image_dir, image_file)

      # Get Attributes for each image 
      value = extract_mat_contents(mat_path, img_path)

      # Append the attributes to the mat_list
      mat_list.append(value)

  # Columns for Pandas DataFrame
  column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 
                 'xmax', 'ymax']

  # Create the DataFrame from mat_list
  mat_df = pd.DataFrame(mat_list, columns=column_name)

  # Return the dataframe
  return mat_df

"""
**Here we're creating a helper function that will convert imagaug's
bounding box objects to panda's dataframe.**
"""

# Function to convert bounding box image into DataFrame 
def bounding_boxes_to_df(bounding_boxes_object):

    # Convert Bounding Boxes Object to Array
    bounding_boxes_array = bounding_boxes_object.to_xyxy_array()
    
    # Convert the array into DataFrame
    df_bounding_boxes = pd.DataFrame(bounding_boxes_array, 
                                     columns=['xmin', 'ymin', 'xmax', 'ymax'])
    
    # Return the DataFrame
    return df_bounding_boxes



"""
This function will finally perform the image augmentations to both images and bounding boxes. It will read the labels dataframe we created earlier to get the bounding box information for each image and as it augments it, and then will also edit the bounding box coordinates so that the coordinates remain true even after augmentation is done.
"""

def image_aug(df, images_path, aug_images_path, augmentor, multiple = 3):
    
    # Fill this DataFrame with image attributes
    augmentations_df = pd.DataFrame(
        columns=['filename','width','height','class', 'xmin', 'ymin', 'xmax',
                 'ymax'])
    
    # Group the data by filenames
    grouped_df = df.groupby('filename')

    # Create the directory for all augmentated images
    if not os.path.exists(aug_images_path):
      os.mkdir(aug_images_path)

    # Create directories for each class of augmentated images
    for folder in df['class'].unique():
      if not os.path.exists(os.path.join(aug_images_path, folder)):
        os.mkdir(os.path.join(aug_images_path, folder))

    for i in range(multiple):
      
      # Post Fix we add to the each different augmentation of one image
      image_postfix = str(i)

      # Loop to perform the augmentations
      for filename in df['filename'].unique():

        augmented_path = os.path.join(aug_images_path, filename)+image_postfix+'.jpg'

        # Take one image at a time with its information
        single_image = grouped_df.get_group(filename)
        single_image = single_image.reset_index()
        single_image = single_image.drop(['index'], axis=1)   
        
        # Read the image
        image = imageio.imread(os.path.join(images_path, filename))

        # Get bounding box
        bounding_box_array = single_image.drop(['filename', 'width', 'height',
                                                'class'], axis=1).values

        # Give the bounding box to imgaug library
        bounding_box = BoundingBoxesOnImage.from_xyxy_array(bounding_box_array, 
                                                            shape=image.shape)

        # Perform random 2 Augmentations
        image_aug, bounding_box_aug = augmentor(image=image, 
                                                bounding_boxes=bounding_box)
        
        # Discard the the bounding box going out the image completely   
        bounding_box_aug = bounding_box_aug.remove_out_of_image()

        # Clip the bounding box that are only partially out of th image
        bounding_box_aug = bounding_box_aug.clip_out_of_image()

        # Get rid of the the image if bounding box was discarded  
        if re.findall('Image...', str(bounding_box_aug)) == ['Image([]']:
            pass
        
        else:
        
          # Create the augmented image file
          imageio.imwrite(augmented_path, image_aug) 

          # Update the image width and height after augmentation
          info_df = single_image.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)    
          for index, _ in info_df.iterrows():
              info_df.at[index, 'width'] = image_aug.shape[1]
              info_df.at[index, 'height'] = image_aug.shape[0]

          # Add the prefix to each image to differentiate if required
          info_df['filename'] = info_df['filename'].apply(lambda x: x + image_postfix + '.jpg')

          # Create the augmented bounding boxes dataframe 
          bounding_box_df = bounding_boxes_to_df(bounding_box_aug)

          # Concatenate the filenames, height, width and bounding boxes 
          aug_df = pd.concat([info_df, bounding_box_df], axis=1)

          # Add all the information to augmentations_df we initialized above
          augmentations_df = pd.concat([augmentations_df, aug_df])            
      
    # Remove index
    augmentations_df = augmentations_df.reset_index()
    augmentations_df = augmentations_df.drop(['index'], axis=1)

    # Return the Dataframe
    return augmentations_df


"""
Now we will start converting class labels to one hot encoded vectors and the
bounding boxes will be scaled to range 0-1 because it’s easier for the network
to predict values in a fixed range for each image, plus we will be using the
sigmoid activation function for the localization / regression head, and sigmoid
outputs values between 0-1. We will also preprocess the image by resizing it
to a fixed size, converting to RGB channel format and normalizing it by dividing 
t by 255.0. This will help the network in training.
"""

def preprocess_dataset(image_folder, classes_list, df, image_size = 300,):


  # Lists that will contain the whole dataset
  labels = []
  boxes = []
  img_list = []

  # Get height and width of each image in the datafame
  h = df['height']
  w = df['width']

  # Create a copy of the labels in the dataframe
  labels = list(df['class'])

  # Create a copy of the bounding box values and also normalize them 
  for x1, y1, x2, y2 in zip(list(df['xmin']/w), list(df['ymin']/h), 
                            list(df['xmax']/w), list(df['ymax']/h)):
    
    arr = [x1, y1, x2, y2]
    boxes.append(arr)

  # We loop over each class and its labels 
  for class_folder in classes_list:  

    # Set our images directory
    image_dir = os.path.join(image_folder, class_folder)

    # Annotation and Image files
    img_files = sorted(os.listdir(image_dir))

    # Loop over each of the image and its label
    for image_file in img_files:

      # Full path Image
      img_path = os.path.join(image_dir, image_file)

      # Read the image
      img  = cv2.imread(img_path)

      # Resize all images to a fix size
      image = cv2.resize(img, (image_size, image_size))

      # Convert the image from BGR to RGB as NasNetMobile was trained on RGB images
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      # Normalize the image by dividing it by 255.0 
      image = image.astype("float") / 255.0

      # Append it to the list of images
      img_list.append(image)

  return labels, boxes, img_list    


"""
Construct the Multi Output Model
Now it’s time to create our Multi Output model. We won’t be training from
scratch but we will use transfer learning. The model we’ll use is NasnetMobile.
This is a really efficient model with a good balance of speed and accuracy,
It was created through Neural Architecture Research (NAS) which is an emerging
field of AutoML. We’ll first download the model but exclude the top, since
we’ll be adding our own custom top.
"""

def create_model(no_of_classes):
    
    image_size = 300
    # Load the NasNetMobile Model, make sure to exclude the top for transfer learning
    N_mobile = tf.keras.applications.NASNetMobile( input_tensor = Input(
    shape = (image_size, image_size, 3)), 
    include_top=False, 
    weights='imagenet')

    # Freeze the whole model
    N_mobile.trainable = False
    
    # Start by taking the output feature maps from NASNETMobile
    base_model_output = N_mobile.output
    
    # Convert to a single-dimensional vector by Global Average Pooling.

    # We could also use Flatten()(x) but GAP is more effective, it reduces 
    # Parameters and controls overfitting.
    flattened_output = GlobalAveragePooling2D()(base_model_output)

    # Create our Classification Head, final layer contains 
    # Ouput units = no. classes
    class_prediction = Dense(256, activation="relu")(flattened_output)
    class_prediction = Dense(128, activation="relu")(class_prediction )
    class_prediction = Dropout(0.2)(class_prediction)
    class_prediction = Dense(64, activation="relu")(class_prediction)
    class_prediction = Dropout(0.2)(class_prediction )
    class_prediction = Dense(32, activation="relu")(class_prediction)
    class_prediction = Dense(no_of_classes, activation='softmax',
                             name="class_output")(class_prediction)

    # Create Our Localization Head, final layer contains 4 nodes for x1,y1,x2,y2
    # Respectively.
    box_output = Dense(256, activation="relu")(flattened_output)
    box_output = Dense(128, activation="relu")(box_output)
    box_output = Dropout(0.2)(box_output )

    box_output = Dense(64, activation="relu")(box_output)
    box_output = Dropout(0.2)(box_output )

    box_output = Dense(32, activation="relu")(box_output)
    box_predictions = Dense(4, activation='sigmoid',
                            name= "box_output")(box_output)

    # Now combine the two heads
    model = Model(inputs=N_mobile.input, outputs= [box_predictions, 
                                                   class_prediction])

    return model



    
if __name__ == '__main__':
    print('dont run a function file')