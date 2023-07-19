# Master-Thesis-Free-Space-Detetction-for-Autonomous-Driving-using-Deep-Learning

- In this work, the network is tasked with detecting free road surfaces in multiple lanes (ego lane and other lanes) given a single 360 x 640 resolution camera frame.
- The network detects the free road surfaces if it contains a part of the road that the vehicle can safely drive on.
- In particular, the detected road should not contain an obstacle e.g. another vehicle or surfaces that the vehicle cannot drive on, e.g. sidewalks.
- With an additional branch in the network (Multi-Task Network), we can predict the road type (e.g. highway, residential, city street, etc.) together with drivable regions.
- This results in multi-task learning where we do semantic segmentation (i.e. classify each pixel of an image into 3 different
classes i.e. ego lane, other lanes, and background) and multi-class classification simultaneously.
- In the end, the best performing model is deployed on our real test vehicle (Volkswagen Passat) to retrieve the real-time inferences.

Dataset Used: [Berkely Deep Drive (BDD)](https://bdd-data.berkeley.edu/) drivable area dataset.

## Inference example from the trained model took on 1 frame:
![bdd_inference](https://github.com/sachinsharma9780/Master-Thesis-Free-Space-Detetction-for-Autonomous-Driving-using-Deep-Learning/assets/40523048/ae4bb78e-b84b-4e45-87cd-e690a56737b2)


Requirements:

1) Pytorch-Cuda (version > 1.0)
2) Basic Machine learning libraries

## Instructions to run the training script:

1) Use the train.py file to start the training. In that file, you need to mention where to store all the checkpoints and training, validation graphs, and path to the BDD dataset.
2) After that use the command CUDA_VISIBLE_DEVICES= gpu# python train.py to run the training script.

## Instructions to run the Inference script:

1) Inference scripts are stored in the inference_script folder.
2) While inferencing from a specific model e.g. skip connections model you need to import that particular model in the inference script.
3) To run the inference, pass the input image path in the inference script and the output path to store the result. 
4) command e.g.: CUDA_VISIBLE_DEVICES= gpu# python erfnet_inference.py
