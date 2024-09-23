# Deep Learning Courseworks
Project Overview

This repository contains the deep learning projects I developed as part of my postgraduate studies in Deep Learning. The projects focus on applying deep learning techniques using PyTorch to solve real-world problems, ranging from image classification using Convolutional Neural Networks (CNNs) to generating realistic data through Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs).

In both courseworks, I implemented various architectures, conducted extensive hyperparameter tuning, and performed experiments to identify the best-performing models for the specific tasks.

Coursework 1: Convolutional Neural Networks (CNNs)

Problem

The first coursework focuses on image classification using Convolutional Neural Networks. The objective is to classify images into different categories by leveraging the power of CNNs to automatically learn spatial hierarchies of features from images.
Key Features

    Model Architecture: Implemented a deep CNN architecture with layers such as convolutional, pooling, and fully connected layers.
    Regularization Techniques: Applied techniques like dropout and batch normalization to improve generalization.
    Data Augmentation: Used image augmentation strategies like rotation, flipping, and cropping to artificially expand the training dataset and enhance model robustness.
    Hyperparameter Tuning: Conducted experiments with different learning rates, batch sizes, and optimizer choices (e.g., SGD, Adam) to find the optimal configuration for the model.
    Evaluation Metrics: Monitored training/validation accuracy and loss during training and used metrics like precision, recall, and F1-score to evaluate the final model performance.
Results

Through careful tuning and experimentation, the CNN model achieved high accuracy on the classification task, demonstrating the effectiveness of deep learning for image recognition.

Coursework 2: Generative AI (VAE and GAN)

Problem

The second coursework focuses on Generative AI, specifically Variational Autoencoders (VAE) and Generative Adversarial Networks (GANs). The goal is to generate realistic synthetic data by learning the underlying distribution of the dataset.
Key Features

    VAE Implementation: Built a VAE model to learn a probabilistic latent space representation of the data and generate new samples by sampling from this latent space.
    GAN Implementation: Developed a GAN consisting of a generator and a discriminator network to generate realistic images. The generator learns to produce convincing fake images while the discriminator learns to distinguish between real and generated images.
    Training Techniques: Used advanced techniques like label smoothing, gradient penalty, and latent space interpolation to stabilize the training of the GAN and improve generation quality.
    Hyperparameter Tuning: Conducted extensive tuning of the learning rates, latent dimensions, batch sizes, and the architecture of the generator and discriminator networks.
    Evaluation Metrics: Evaluated the quality of generated images using qualitative assessments (visual inspection) and quantitative metrics like Fréchet Inception Distance (FID) to measure how close the generated images are to the real dataset.

Results

The trained GAN model was able to generate visually convincing samples, while the VAE effectively learned the latent representation of the data, allowing smooth interpolations between different generated images.
