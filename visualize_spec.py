import matplotlib.pyplot as plt
import os
import numpy as np
import librosa
import librosa.display

gen_file_path = "./samples/generated/"
og_file_path = "./samples/original/"
single_file_path = ""
spec_comparison_save_path = "./samples/spec_comparison/"

fp = "samples/original/"



HOP_LENGTH = 256
FRAME_SIZE = HOP_LENGTH * 4

def visualize_single_spec(file):
    #------------------Load & Convert Spectrogram----------------#
    audio_load, sr = librosa.load(file)
    audio_signal = librosa.stft(audio_load, n_fft = FRAME_SIZE, hop_length= HOP_LENGTH)

    #------------------Calculate the Spectrogram------------------#
    #------Shape remains, values change from complex to real------#

    Y_scale = np.abs(audio_signal) ** 2     #shows an empty spectrogram, we need a log-spec
    Y_log_scale = librosa.power_to_db(Y_scale)
    #-------------------Plot the Spectrogram-----------------------#
    plot_single_spectrogram(Y_log_scale,sr,HOP_LENGTH)

def visualize_single_mel_spec(file):

    audio_load, sr = librosa.load(file)
    #filter_banks = librosa.filters.mel(n_fft=FRAME_SIZE,sr=44100,n_mels=128)
    mel_spec = librosa.feature.melspectrogram(audio_load, sr = sr, n_fft= FRAME_SIZE, hop_length=HOP_LENGTH, n_mels = 512)
    log_mel_spec = librosa.power_to_db(mel_spec)
    librosa.display.specshow(log_mel_spec,
                             x_axis = "time",
                             y_axis = "mel",
                             sr = sr)
    plt.colorbar(format="%+2.f")
    plt.show()

def compare_spec(g_path,og_path, path_index):
    #------------------Load & Convert Mel Spectrogram----------------#
    g_audio, sr = librosa.load(g_path)
    og_audio, sr = librosa.load(og_path)
    g_audio_signal = librosa.feature.melspectrogram(g_audio, sr = sr, n_fft = FRAME_SIZE, hop_length= HOP_LENGTH, n_mels = 512)
    og_audio_signal = librosa.feature.melspectrogram(og_audio, sr = sr, n_fft = FRAME_SIZE, hop_length= HOP_LENGTH, n_mels = 512)

    g_log_scaled = librosa.power_to_db(g_audio_signal)
    og_log_scaled = librosa.power_to_db(og_audio_signal)

    #-------------------Plot the Spectrogram-----------------------#
    plot_comparison_spectrogram(g_log_scaled, og_log_scaled, sr, HOP_LENGTH, path_index)

def compare_spec_old(g_path,og_path, path_index):
    #------------------Load & Convert Spectrogram----------------#
    g_audio, sr = librosa.load(g_path)
    og_audio, sr = librosa.load(og_path)
    g_audio_signal = librosa.stft(g_audio, n_fft = FRAME_SIZE, hop_length= HOP_LENGTH)
    og_audio_signal = librosa.stft(og_audio, n_fft = FRAME_SIZE, hop_length= HOP_LENGTH)

    #------------------Calculate the Spectrogram------------------#
    #------Shape remains, values change from complex to real------#

    g_scaled = np.abs(g_audio_signal) ** 2
    g_log_scaled = librosa.power_to_db(g_scaled)
    og_scaled = np.abs(og_audio_signal) ** 2
    og_log_scaled = librosa.power_to_db(og_scaled)

    #-------------------Plot the Spectrogram-----------------------#
    plot_comparison_spectrogram(g_log_scaled, og_log_scaled, sr, HOP_LENGTH, path_index)



def plot_single_spectrogram(Y, sr, hop_length, y_axis = "log"):
    plt.subplot(211)
    librosa.display.specshow(Y,
                             sr=sr,
                             hop_length=hop_length,
                             x_axis="time",
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")
    plt.savefig(spec_comparison_save_path + "0.png")
    plt.show()

def plot_comparison_spectrogram(generated, original, sr, hop_length, path_index = "000", y_axis = "log" ):
    plt.subplot(211).set_title("Generated")
    librosa.display.specshow(generated,
                             sr=sr,
                             hop_length=hop_length,
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")

    plt.subplot(212).set_title("Original")
    librosa.display.specshow(original,
                             sr=sr,
                             hop_length=hop_length,
                             x_axis="time",
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")
    plt.savefig(spec_comparison_save_path + str(path_index) + ".png")
    #plt.show()

def get_length_of_folder(path):
    i = 0
    for root, dirs, files in os.walk(path):
        for name in files:
            i+= 1
    return i

def compare_spec_in_folder(length):

    for i in range(length-1):
        gen_path = gen_file_path + str(i) + ".wav"
        og_path = og_file_path + str(i) + ".wav"
        compare_spec(gen_path,og_path, path_index = i)
        print(f"{i}.wav save completed!")



#--------Next Step, Plot OG and Reconstructed Spec for Analysis, and also Latent One using subplots -----#



#if __name__ == "__main__":
    #visualize_single_spec(single_file_path)
    #visualize_single_mel_spec(single_file_path)
    #compare_spec(gen_file_path, og_file_path)
    #compare_spec_in_folder(get_length_of_folder(fp))

