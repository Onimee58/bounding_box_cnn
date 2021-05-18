# bounding_box_cnn
recognition and localization using bounding box and custom Convolutional Neural Networks

Download the caltech101 dataset from : https://zenodo.org/record/4126613/files/CALTECH
# Some images from the datasets with bbox drawn in them
![image](https://user-images.githubusercontent.com/50075168/118710427-14b8b880-b840-11eb-9be7-be8af2ffa98d.png)

The model is trained o CALTECH101 dataset whish has 3 classes namely elephant, butterfly, and cougar-face. The class labels and bounding box coordinates for each image are present in .MAT (Matlab) format.

1) run predict.py to check the performance on pretrained model
2) or run train.py to train the model
3) here following part was added to the head

![image](https://user-images.githubusercontent.com/50075168/118710514-36b23b00-b840-11eb-92f4-7b54de90d3ea.png)

4) make your own head in the function_needed.py file.


