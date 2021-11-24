
import os
import pickle

import numpy as np
from autoencoder import VAE


LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 100
SPECTROGRAMS_PATH = "/Users/nicks./Documents/datasets/fsdd/spectrograms_lux"

def load_fsdd(spectrogram_path):
    x_train = []
    for root, _, file_names in os.walk(spectrogram_path):
        for file_name in file_names:
            file_path = os.path.join(root,file_name)
            spectrogram = np.load(file_path, allow_pickle=True) # (n_bins,n_frames)
            x_train.append(spectrogram)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # -> (3000, 256, 64, 1) treating spectrograms as greyscale
    return x_train


def train(x_train, learning_rate, batch_size, epochs):
    vae = VAE(
        input_shape=(256, 128, 1),     #256,528,1 -> 3 secs
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2,1)),
        latent_space_dim= 64 #original 128
    )
    vae.summary()
    vae.compile(learning_rate)
    vae.train(x_train, batch_size, epochs)
    return vae


if __name__ == "__main__":
    x_train = load_fsdd(SPECTROGRAMS_PATH)
    vae = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    vae.save("model")