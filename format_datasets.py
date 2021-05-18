 # -*- coding: utf-8 -*-
"""
Created on Tue May 18 23:24:26 2021

@author: Saif
"""

from imgaug import augmenters as iaa 
from needed_functions import mat_to_csv, image_aug, preprocess_dataset
import cv2
import random
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
   
def format_datasets():
    # The Classes we will use for our training
    classes_list = sorted(['butterfly',  'cougar_face', 'elephant'])
    
    
    # Set our images and annotations directory
    image_directory = 'CALTECH/CALTECH_Dataset'
    annot_directory = 'CALTECH/CALTECH_Annotations'
    
    # Run the function to convert all the MAT files to a Pandas DataFrame
    labels_df = mat_to_csv(annot_directory, image_directory, classes_list)
    
    # Saving the Pandas DataFrame as CSV File
    labels_df.to_csv(('labels.csv'), index=None)
    
    """Now that we have the data ready in the required format we can proceed to perform the augmentations
    
    """
    
    # Define all the Augmentations you want to apply to your dataset
    # We're setting random `n` agumentations to 2. 
    image_augmentations = iaa.SomeOf( 2,
        [                                 
        # Scale the Images
        iaa.Affine(scale=(0.5, 1.5)),
    
        # Rotate the Images
        iaa.Affine(rotate=(-60, 60)),
    
        # Shift the Image
        iaa.Affine(translate_percent={"x":(-0.3, 0.3),"y":(-0.3, 0.3)}),
    
        # Flip the Image
        iaa.Fliplr(1),
    
        # Increase or decrease the brightness
        iaa.Multiply((0.5, 1.5)),
    
        # Add Gaussian Blur
        iaa.GaussianBlur(sigma=(1.0, 3.0)),
        
        # Add Gaussian Noise
        iaa.AdditiveGaussianNoise(scale=(0.03*255, 0.05*255))
    
    ])
    
    
    
    augmented_images_df = image_aug(labels_df, image_directory, 'aug_images', 
                                    image_augmentations)
    
    """Apply the function to do augmentation"""
    
    augmented_images_df = augmented_images_df.sort_values('filename', ignore_index= True)
    augmented_images_df.to_csv('aug.csv')
    
    # Check Dataset Size
    print('Our total dataset Size before the augmentations was: ', len(labels_df))
    print('Our total dataset Size after the augmentations is: ', len(augmented_images_df))
    
    
    """Now call the preprocessing function"""
    
    # All images will resized to 300, 300 
    image_size = 300
    
    # Get Augmented images and bounding boxes
    labels, boxes, img_list = preprocess_dataset('aug_images', classes_list,
                                                 augmented_images_df)
    
    """We'll Shuffle the data after preprocessing is done."""
    
    # Convert labels to integers, then one hot encode them
    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(labels)
    onehot_labels = to_categorical(integer_labels)
    
    # Now we need to shuffle the data, so zip all lists and shuffle
    combined_list = list(zip(img_list, boxes, onehot_labels))
    random.shuffle(combined_list)
    
    # Extract back the contents of each list
    img_list, boxes, onehot_labels = zip(*combined_list)
    
    print('All Done')
    
    """
    Visualize the Data with its Annotations**
    Here we will pick some random images from each class and draw its 
    associated bounding boxes over it. This way you’ll be able to visualize
    the data and make sure that your preprocessing steps were done correctly.
    """
    
    # Create a Matplotlib figure
    plt.figure(figsize=(20,20));
    
    # Generate a random sample of images each time the cell is run 
    random_range = random.sample(range(1, len(img_list)), 20)
    
    for iteration, i in enumerate(random_range, 1):
    
        # Bounding box of each image
        a1, b1, a2, b2 = boxes[i];
    
        # Rescaling the boundig box values to match the image size
        x1 = a1 * image_size
        x2 = a2 * image_size
        y1 = b1 * image_size
        y2 = b2 * image_size
    
        # The image to visualize
        image = img_list[i]
    
        # Draw bounding boxes on the image
        cv2.rectangle(image, (int(x1),int(y1)),
              (int(x2),int(y2)),
                      (0,255,0),
                      3);
        
        # Clip the values to 0-1 and draw the sample of images
        image = np.clip(img_list[i], 0, 1)
        plt.subplot(4, 5, iteration);
        plt.imshow(image);
        plt.axis('off');
    
    """
    Split the Data into Train and Validation Set**
    Now we would have 3 lists, one list containing all images, the second one 
    contains all class labels in one hot encoded format and the third one list 
    contains scaled bounding box coordinates. Let’s split our data to create a
    training and validation set. It’s important that you shuffle your data before
     the split which we have already done.
    """
    
    # Split the data of images, labels and their annotations
    train_images, val_images, train_labels, val_labels, train_boxes, val_boxes = train_test_split( np.array(img_list),
                                                                                                  np.array(onehot_labels),
                                                                                                  np.array(boxes),
                                                                                                  test_size = 0.1,
                                                                                                  random_state = 43)
    
    print('Total Training Images: {}, Total Validation Images: {}'.format(
        len(train_images), len(val_images) ))
    
    return train_images, val_images, train_labels, val_labels, train_boxes, val_boxes

if __name__ == '__main__':
    print('dont run a function file')