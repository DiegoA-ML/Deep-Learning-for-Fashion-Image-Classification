# Deep-Learning-for-Fashion-Image-Classification

## Overview: 

This project focused on developing and evaluating deep learning models to classify grayscale images of fashion items from the Fashion-MNIST dataset. We aimed to explore different neural network architectures and optimize them for performance and efficiency.

## Objectives:

 1 Implement and compare different Convolutional Neural Network (CNN) architectures.

 2 Evaluate the performance of EfficientNet, a scalable and efficient neural network, on Fashion-MNIST.
 
 3 Analyze model accuracy, training time, and memory usage to identify the best model for this classification task.

## Key Technologies:

 • Python
 
 • TensorFlow / Keras
 
 • Fashion-MNIST dataset

## Methodology:
 
### 1 Data Preprocessing:
 
 ◦ Normalized image pixel values to [0, 1].
 
 ◦ Applied one-hot encoding to class labels.
 
### 2 Implemented Models:
 
 ◦ LeNet-5: Achieved 89% accuracy.
 
 ◦ Jason Brownlee’s CNN Tutorial Model: Achieved 88% accuracy.
 
 ◦ Custom CNN: Achieved 92% accuracy by incorporating dropout layers and batch normalization.
 
 ◦ EfficientNet: Fine-tuned for Fashion-MNIST, achieving 91.9% accuracy.
 
### 3 Evaluation:
 
 ◦ Accuracy, loss, and confusion matrices were analyzed for each model.
 
 ◦ EfficientNet demonstrated superior performance in terms of scalability and generalization.

## Challenges:

 • Managing overfitting in CNN models.
 
 • Optimizing hyperparameters such as learning rate and dropout ratio.

## Results:

 • The Custom CNN achieved the highest accuracy (92%), closely followed by EfficientNet (91.9%).
 
 • EfficientNet showcased its efficiency by maintaining high accuracy with lower memory usage (~20 MB) compared to CNN models.

## Future Work:
 
 • Implement advanced data augmentation techniques to further improve model performance.
 
 • Explore transfer learning using pre-trained models like those trained on ImageNet for better generalization.

Skills: Data Science · Deep Learning · Convolutional Neural Networks (CNN) · Predictive Modeling · TensorFlow
