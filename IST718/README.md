# IST718
This project is based on the Extended MNIST data set of images of handwritten letters and and numbers. The aim was to develop machine learning
and deep learning model to identify the characters in the images. A secondary objective was to compare the performances by tuning the hyper-
parameters. originally, me and the team decided to work with the Stanford house numbers data set. The problem was that there were about 
1.2 million images in that dataset and each images waas 128 pixels*128 pixels. It crashed the Apache Spark network we were using as we ran 
out of computation resources over the network before the task was finished. Hence, we switched to EMNIST. It had about 600k images and every
image was 28 pixels*28 pixels. After reading the images, reduction of dimensionality was doen by using Principal Component Analysis. PCA helps
a lot, it made the ML and DL parts less complicated and resource consuming than what it'd have had been without PCA. Multi-layer perceptron
outperformed Random Forest. However, Random Forest takes about 10-12 minutes to be trained and ready for testing. Whereas, it takes around
45 minutes for MLP to form and back-propogate and be ready for testing.
