# MarineNN: Underwater Image Enhancement

The MarineNN capstone project is a deep learning-based solution designed to address the challenges associated with underwater image enhancement, focusing on real-time color correction and dehazing. The objective is to develop a robust and efficient model that can operate on systems with limited hardware resources, making it accessible to a wide range of users and applications.

## Project Goals

The project's primary goals are:

1. Design and implement a deep learning model, called **MarineNN**, for underwater image enhancement that can handle color correction, dehazing, and preserving fine details.
2. Create a user-friendly tool that is easily adaptable for individuals and organizations, regardless of their expertise or available resources.
3. Evaluate the performance of MarineNN using diverse underwater image datasets, such as the University of Minnesota's Enhancing Underwater Visual Perception (EUVP) collection, to ensure its generalizability.

## Approach

To achieve these goals, the project employs state-of-the-art deep learning techniques, including the use of convolutional neural networks (CNNs) and skip connections. The model architecture is inspired by the U-Net, which has demonstrated success in various image processing tasks. A novel loss function, combining SSIM and MS-SSIM, is employed to optimize the model during training, ensuring high-quality image enhancement while minimizing artifacts.

## Results

The MarineNN model is extensively evaluated using a diverse set of underwater images, demonstrating its effectiveness in improving image quality, preserving details, and outperforming existing solutions, such as the UGAN (Underwater Generative Adversarial Network).

## Conclusion

In conclusion, the MarineNN capstone project successfully addresses the need for efficient and effective underwater image enhancement. It has the potential to impact various underwater applications, including marine biology research, environmental monitoring, and recreational diving, by providing enhanced images for better visual perception and analysis.

## Dataset

Access the dataset used in this project [here](https://drive.google.com/drive/folders/1ZEql33CajGfHHzPe1vFxUFCMcP0YbZb3?usp=sharing).


## How to Run the Project

Follow the steps below to execute the code files for this project:

1. **dataloader.py**
Prepares the data and creates the data loaders for training and validation.

2. **TRAINING_CONFIG.py**
Sets the configuration parameters for training.

3. **training.py**
Trains the model using the data loaders and configuration parameters.

4. **test.py**
Tests the trained model on a set of test images.

5. **converting_video.py** (Optional)
After training and testing the model, you can use this script to convert a video into frames and process them with the trained model.

> **Note**: `loss.py` and `model.py` do not need to be executed separately, as they are imported and used in the `training.py` and `test.py` scripts.
