# -*- coding: utf-8 -*-
"""
Created on Tue May 18 23:24:26 2021

@author: Saif
"""


import tensorflow as tf
tf.random.set_seed(100)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import  EarlyStopping, ReduceLROnPlateau
from keras.utils.vis_utils import plot_model
from needed_functions import create_model
import matplotlib.pyplot as plt
import imgaug as ia
ia.seed(1)
from format_datasets import format_datasets



classes_list = sorted(['butterfly',  'cougar_face', 'elephant'])
train_images, val_images, train_labels, val_labels, train_boxes, val_boxes = format_datasets()

model = create_model(len(classes_list))
print("Model Created")

"""
Using the plot_model function you can check the structure of the final model,
this is really helpful when you’re creating a complex network and you want
to make sure you have constructed the network correctly.
"""

plot_model(model, to_file='model_plot.png', show_shapes=True, 
           show_layer_names=True)

"""
Compile & Train the Model**
Now here are a couple of things that you need to do. If this was a single
output model you would have just defined a loss, a metric and an optimizer 
and compiled the model. Since we’re dealing with a multi output model that 
outputs two totally different things so we need to use seperate loss and 
separate metrics for both of these output branches. Remember when we were 
creating the model we actually assigned names to our two output branches by 
doing that now we can easily refer to the name of an output branch and then 
assign it a specific loss or metric, just like a python dictionary. 
"""

# Here for each head we will define a different loss, we will define it 
# Like a dictionary.

# For classification we will have cateogirical crossentropy
# For the bouding boxes we will have mean squared error
losses = { 
    "box_output": "mean_squared_error",
    "class_output": "categorical_crossentropy"
    }

# Here you can give more or less weightage to each loss. 

# If you think that detection is harder then the classification then you can 
# Try assinging it more weight
loss_weights = {
    "box_output": 1.0, 
    "class_output": 1.0
    }

# Set the Metrics
# For the class labels we want to know the Accuracy
# And for the bounding boxes we need to know the Mean squared error
metrics = {
    'class_output': 'accuracy', 
    'box_output':  'mse'
    }

# We will be using early stopping to stop the model if total val loss does not
# Decrease by 0.001 in 40 epochs
stop = EarlyStopping(monitor = "val_loss", min_delta = 0.0001, patience = 40, 
                    restore_best_weights = True
                     )

# Change the learning rate according to number of epochs to boost learning
reduce_lr = ReduceLROnPlateau(monitor = "val_loss", factor = 0.0002, 
                              patience = 30, min_lr = 1e-7, verbose = 1)

# Initialize Optimizer
opt = SGD(lr = 1e-3, momentum = 0.9)
opt2 = Adam

# Compile the model with Adam optimizer
model.compile(optimizer = opt, loss = losses, loss_weights = loss_weights, 
    metrics = metrics)

"""
Start Training:**
When you're dealing with a multi output model then you need to assingn
individual branches labels seperately like a dictionary. 
"""

# Train the Model
history = model.fit(x = train_images, 
                    y= {
                        "box_output": train_boxes, 
                        "class_output": train_labels
                        }, 
                    validation_data=(
                        val_images, 
                        {
                          "box_output": val_boxes, 
                          "class_output": val_labels
                          }), batch_size = 32, epochs = 500, 
                    callbacks=[reduce_lr, stop])

"""
plot Model’s Loss & Accuracy Curves**
Now we had seperate losses and sperate metrics for differnt branches,
on top of that we used a validation set. We're now dealing with 10 total
metrics. Let's visualize and understand these by plotting them.
"""

def eval_plot(var1, var2, plot_name):
  # Get the loss metrics from the trained model
  c1 = history.history[var1]
  c2 = history.history[var2]

  epochs = range(len(c1))
  
  # Plot the metrics
  plt.plot(epochs, c1, 'b', label=var1)
  plt.plot(epochs, c2, 'r', label=var2)
  plt.title(str(plot_name))
  plt.legend()

eval_plot('class_output_accuracy','val_class_output_accuracy','Class Output Accuracy vs Class Validation Accuracy')

eval_plot('class_output_loss','val_class_output_loss','Class Output Loss vs Class Validation Loss')

eval_plot('box_output_mse','val_box_output_mse','Box Output MSE vs Box Validation MSE')

eval_plot('box_output_loss','val_box_output_loss','Box Output Loss vs Box Validation Loss')

# These are the most important metrics in telling us how our model is doing
eval_plot('loss','val_loss',' Total Loss vs Total Validation Loss')

"""
Save Your Model**
You should save your model for future runs.
"""

# Save your model here in .h5 format.
model.save('caltech.h5')



