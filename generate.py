import os
import pickle

import numpy as np
import soundfile as sf
import visualize_spec as spec
from soundgenerator import SoundGenerator
from autoencoder import VAE




HOP_LENGTH = 256
SAVE_DIR_ORIGINAL = "samples/original/"
SAVE_DIR_GENERATED = "samples/generated/"
SAVE_DIR_LATENT = "samples/latent/"
MIN_MAX_VALUES_PATH = "./datasets/min_max_values.pkl"
SPECTROGRAMS_PATH = "./datasets/mel_spectrograms"



def load_fsdd(spectrograms_path):
    x_train = []
    file_paths = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in [i for i in file_names if not (i.startswith('.'))]:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path, allow_pickle=True) # (n_bins, n_frames, 1)
            x_train.append(spectrogram)
            file_paths.append(file_path)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # -> (no. samples, 256, 64, 1)
    return x_train, file_paths




def select_spectrograms(spectrograms,
                        file_paths,
                        min_max_values,
                        num_spectrograms=5):
    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms)
    sampled_spectrograms = spectrograms[sampled_indexes]
    file_paths = [file_paths[index] for index in sampled_indexes]
    sampled_min_max_values = [min_max_values[file_path] for file_path in file_paths]
    print(file_paths)
    print(sampled_min_max_values)
    print("the type of sampled specs" ,type(sampled_spectrograms))
    return sampled_spectrograms, sampled_min_max_values


def save_signals(signals, save_dir, sample_rate=44100):
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, str(i) + ".wav")
        sf.write(save_path, signal, sample_rate)


if __name__ == "__main__":
    # initialise sound generator
    vae = VAE.load("model")
    sound_generator = SoundGenerator(vae, HOP_LENGTH)

    # load spectrograms + min max values
    with open(MIN_MAX_VALUES_PATH, "rb") as f:
        min_max_values = pickle.load(f)

    specs, file_paths = load_fsdd(SPECTROGRAMS_PATH)

    # sample spectrograms + min max values
    sampled_specs, sampled_min_max_values = select_spectrograms(specs,
                                                                file_paths,
                                                                min_max_values,
                                                                10)

    # generate audio for sampled spectrograms
    signals, _ = sound_generator.generate(sampled_specs,
                                          sampled_min_max_values)
    # generate random points
    sampled_points = []
    for i in range(10):
       z = vae.generate_random_point_latent_space()
       sampled_point = np.array(vae.sample_from_latent_space(z))
       new_signal = sound_generator.convert_spectrograms_to_audio(sampled_point, sampled_min_max_values)
       sampled_points.append(new_signal[0])

    # convert spectrogram samples to audio
    original_signals = sound_generator.convert_spectrograms_to_audio(
        sampled_specs, sampled_min_max_values)

    # save audio signals
    save_signals(signals, SAVE_DIR_GENERATED)
    save_signals(original_signals, SAVE_DIR_ORIGINAL)
    save_signals(sampled_points, SAVE_DIR_LATENT)
    spec.compare_spec_in_folder(spec.get_length_of_folder(SAVE_DIR_ORIGINAL))