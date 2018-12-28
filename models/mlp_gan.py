from blocks import layers, activations, losses
from utils import load_data, network, optimizers, split
from utils.transforms import *
import numpy as np
import matplotlib.pyplot as plt


# hyper-parameters
d_lr = 0.0002
g_lr = 0.0002
image_size = 28*28
hideen_size = 256
latent_size = 64
batch_size = 100
num_epoches = 400

# discriminator
D = network.Network(l2_reg=1e-3)
D.add_block(layers.Linear(image_size, hideen_size))
D.add_block(layers.BatchNorm1d(hideen_size))
D.add_block(activations.LeakeyReLU(0.2))
D.add_block(layers.Linear(hideen_size, hideen_size))
D.add_block(layers.BatchNorm1d(hideen_size))
D.add_block(activations.LeakeyReLU(0.2))
D.add_block(layers.Linear(hideen_size, 1))
D.add_block(losses.BCELoss())

# generator
G = network.Network(l2_reg=1e-3)
G.add_block(layers.Linear(latent_size, hideen_size))
G.add_block(layers.BatchNorm1d(hideen_size))
G.add_block(activations.ReLU())
G.add_block(layers.Linear(hideen_size, hideen_size))
G.add_block(layers.BatchNorm1d(hideen_size))
G.add_block(activations.ReLU())
G.add_block(layers.Linear(hideen_size, image_size))
G.add_block(activations.Tanh())

# optimizers
d_optimizer = optimizers.Adam(learning_rate=d_lr, betas=(0.5, 0.99))
d_optimizer.load(D.params, D.grads)

g_optimizer = optimizers.Adam(learning_rate=g_lr, betas=(0.5, 0.99))
g_optimizer.load(G.params, G.grads)

# data loading and augmentation
train_transform = Transforms([ToTensor(), Normalize(mean=0.5, std=0.5)])
_ = Transforms([ToTensor()])
train = load_data.load_mnist(train_transform, _)[0]

# denormalization
def denorm(x):
    x = (x + 1) / 2
    x[x > 1] = 1
    x[x < 0] = 0
    return x

def reset_grad():
    D.zero_grad()
    G.zero_grad()

# training
for epoch in range(num_epoches):
    x_batches, _ = split.split_minibatch(train[0], train[1], batch_size, shuffle=True)
    num_iters = len(x_batches)
    for i in range(num_iters):
        size = x_batches[i].shape[0]
        real_labels = np.ones(size, dtype=int)
        fake_labels = np.zeros(size, dtype=int)

        # train the discriminator
        for j in range(1):
            reset_grad()
            d_loss_real = D.forward(x_batches[i].reshape(size, -1), real_labels)
            D.backward()

            z = np.random.randn(size, latent_size)
            fake_images = G.forward(z, None)
            d_loss_fake = D.forward(fake_images, fake_labels)
            D.backward()

            d_optimizer.step(epoch)

        # train the generator
        for k in range(1):
            reset_grad()
            z = np.random.randn(size, latent_size)
            fake_images = G.forward(z, None)
            g_loss = D.forward(fake_images, real_labels)
            grad = D.backward()
            G.backward(grad)
            g_optimizer.step(epoch)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.6f}, g_loss: {:.6f}'
                  .format(epoch, num_epoches, i + 1, num_iters, d_loss_real + d_loss_fake, g_loss))

    if (epoch + 1) % 10 == 0:
        sample = np.random.randn(10, latent_size)
        images = denorm(G.forward(sample, None)).reshape(10, 28, 28)
        fig = plt.figure(figsize=(28, 28))
        for m in range(10):
            fig.add_subplot(1, 10, m+1)
            plt.imshow(images[m], cmap='gray')
            plt.xticks([])
            plt.yticks([])
        plt.show()
