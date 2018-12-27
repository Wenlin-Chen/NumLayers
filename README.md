NumLayers
====

A Deep Learning Library Written in NumPy
----
- A personal project for fun and for the purpose of learning <br>
- Author: Wenlin Chen <br>
- E-mail: chen.wenlin@outlook.com <br><br>
26th December, 2018 <br><br>

Requirements
----
- Python 3.x<br>
- NumPy<br>
- Matplotlib<br><br>

Getting Started
----
- MNIST dataset<br>
Download *http://deeplearning.net/data/mnist/mnist.pkl.gz* <br>
and put it in *./data/mnist/* <br>

- CIFAR10 dataset<br> 
Dwonload *https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz* <br>
extract it and put the folder *cifar-10-batches-py* in *./data/cifar10/*<br>

- Enable Cython acceleration for Conv2d<br>
Run the following command from the *./utils/im2col/* directory:<br>
*python setup.py build_ext --inplace*<br>

- Run the example models in *./models/*<br><br>

Features
----
Layers<br>
- Linear<br>
- Dropout<br>
- Convolution 2D<br>
- Max Pooling 2D<br>
- Batch Normalization 1D & 2D<br>

Losses<br>
- Hinge Loss<br>
- Cross Entropy Loss<br>
- Mean Square Error Loss<br>
- Binary Cross Entropy Loss<br>

Activation Functions<br> 
- Tanh<br>
- ReLU<br>
- Sigmoid<br>
- Leaky ReLU<br>

Optimizers<br>
- Adam<br>
- Momentum and Nesterov<br>
- Stochastic Gradient Descent<br>

Training Tools<br>
- L2 Regularization<br>
- Data Augmentation<br>
- Learning Rate Decay<br><br>

Training Curves
----
- *./models/mlp_mnist.py*<br>
![MLP for MNIST training curve](https://github.com/Wenlin-Chen/NumLayers/blob/master/logs/mlp_mnist.png)<br><br>
- *./models/mlp_cifar10.py*<br>
![MLP for CIFAR10 training curve](https://github.com/Wenlin-Chen/NumLayers/blob/master/logs/mlp_cifar10.png)<br><br>
- *./models/cnn_mnist.py*<br>
![CNN for MNIST training curve](https://github.com/Wenlin-Chen/NumLayers/blob/master/logs/cnn_mnist.png)<br><br>
- *./models/cnn_cifar10.py*<br>
![CNN for CIFAR10 training curve](https://github.com/Wenlin-Chen/NumLayers/blob/master/logs/cnn_cifar10.png)<br><br>