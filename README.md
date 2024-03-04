# MathTexViOCR
This repository contains the code of the MathtexViOCR model. The model is fine-tuned from the pix2tex model on the Vietnamese mathematics data set. The dataset includes 200k images for training and 5000 images for testing. The images in the dataset contain mathematical equations collected from Vietnamese exam questions. The model achieved 84% accuracy on the test set.

# Features

## Training mode
The Repository contains training code and model evaluation. 

## Served model
### Convert model to onnx format
### Serving model by Triton Inference Server