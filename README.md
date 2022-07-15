# Master-Thesis-Free-Space-Detetction-for-Autonomous-Driving-using-Deep-Learning

Given an input image the task is to detect free road areas in all the lanes and classify the given input image into one of classes (city_street, highway, ,etc). This results to a multi-task learning where we do semantic segmentation (i.e classify each pixel of an image into 3 different
classes i.e. ego lane, other lanes and background) and mult-class classification simultaneously.

Requirements:

1) Pytorch-Cuda (version > 1.0)
2) Basic Machine learning libraries

## Instructions to run the training script:

1) Use the train.py file to start the training. In that file, you need to mention where to store all the checkpoints and training, validation graphs, path to the BDD dataset .
2) After that use the command CUDA_VISIBLE_DEVICES= gpu# python train.py to run the training script.

## Instrunctions to run the Inference script:

1) Inference scripts are stored in inference_script folder.
2) While inferencing from specific model e.g skip connections model you need to import that particular model in the inference script.
3) To run the inference, pass the input image path in the inference script and the output path to store the result. 
4) command e.g. : CUDA_VISIBLE_DEVICES= gpu# python erfnet_inference.py
