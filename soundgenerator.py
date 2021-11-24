from preprocess import MinMaxNormaliser
import librosa
import numpy as np

class SoundGenerator:
    """generate audio from spectrograms"""


    def __init__(self, vae, hop_length):
        self.vae = vae
        self.hop_length = hop_length
        self._min_max_normaliser = MinMaxNormaliser(0,1)

    def generate(self, spectrograms, min_max_values):
        generated_spectrograms, latent_representations = self.vae.reconstruct(spectrograms)
        print(f"Shape of generated spectrogram :{generated_spectrograms.shape}")
        signals = self.convert_spectrograms_to_audio(generated_spectrograms, min_max_values)
        return signals, latent_representations


    def convert_spectrograms_to_audio(self, spectrograms, min_max_values):
        signals = []
        for spectrogram, min_max_value in zip(spectrograms, min_max_values):
            # reshape the log spectrogram
            log_spectrogram = spectrogram[:, :, 0]
            # denormalise
            denorm_log_spec = self._min_max_normaliser.denormalise(log_spectrogram, min_max_value["min"], min_max_value["max"])
            # log spectrogram -> spectrogram
            spec = librosa.db_to_power(denorm_log_spec)
            i_signal = librosa.feature.inverse.mel_to_stft(spec, sr = 44100, n_fft = 512)
            # apply stft griffin lim
            signal = librosa.griffinlim(i_signal, window="hann", hop_length=self.hop_length)
            # append signal to "signals"
            signals.append(signal)
        return signals

