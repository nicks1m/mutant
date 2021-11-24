import augly.audio as audaugs
import numpy as np
import librosa
import soundfile as sf
import os

audio_path2 = "/Users/nicks./Documents/datasets/fsdd/data_lux/lux_S1_808_xelth.wav"
audio_path = "/Users/nicks./Documents/datasets/fsdd/softies/3_sec 10-X24.wav"
output_path = "/Users/nicks./Documents/datasets/fsdd/aug/ps3.wav"

data_folder = "/Users/nicks./Documents/datasets/fsdd/softies"
save_folder = "/Users/nicks./Documents/datasets/fsdd/s_aug/"
save_folder_b = "/Users/nicks./Documents/datasets/fsdd/s_aug_b/"
save_folder_c = "/Users/nicks./Documents/datasets/fsdd/s_aug_c/"


def aug_dataset(data_folder, save_folder):
    for root, _, files in os.walk(data_folder):
        for file in [i for i in files if not (i.startswith('.') or i.endswith('.aif'))]:
            file_path = os.path.join(root, file)
            save_path = os.path.join(save_folder, "a_" + file)

            audio_signal, sr = librosa.load(file_path)
            t_signal = apply_augmentation(audio_signal, sr)
            sf.write(save_path, t_signal[0], sr)
            print("augmentation saved!")

#Adding Reverb to Dataset
def add_reverb_dataset(data_folder, save_folder):
    for root, _, files in os.walk(data_folder):
        for file in [i for i in files if not (i.startswith('.') or i.endswith('.aif'))]:
            file_path = os.path.join(root, file)
            save_path = os.path.join(save_folder, "c_" + file)
            sig, sr = librosa.load(file_path)
            audio_signal = audaugs.reverb(audio=sig,output_path=save_path)
            #sf.write(save_path, audio_signal[0], sr)
            print("augmentation saved!")

#First Set of Augmentations on Data
TRANSFORMS = audaugs.Compose([
    audaugs.OneOf([
        audaugs.PitchShift(n_steps=9), audaugs.PitchShift(n_steps=6),
    ]),
    # audaugs.OneOf([
    #    audaugs.HighPassFilter(cutoff_hz=6000.0), audaugs.LowPassFilter(cutoff_hz=300)
    # ]),
    audaugs.OneOf([
        audaugs.Harmonic(power=3.0), audaugs.Percussive(power=3.0),
    ]),
    audaugs.OneOf(
        [audaugs.Speed(factor=0.5), audaugs.TimeStretch(rate=0.5)]
    ),
    audaugs.OneOf([
        audaugs.InvertChannels(), audaugs.ChangeVolume(volume_db=3.0)
    ])
])

#Second Set of Augmentations on Data
TRANSFORMS_B = audaugs.Compose([
    audaugs.OneOf([
        audaugs.PitchShift(n_steps=-3), audaugs.PitchShift(n_steps=-6),
    ]),
    # audaugs.OneOf([
    #    audaugs.HighPassFilter(cutoff_hz=6000.0), audaugs.LowPassFilter(cutoff_hz=300)
    # ]),
    audaugs.OneOf([
        audaugs.Harmonic(power=1.5), audaugs.Percussive(power=3.0),
    ]),
    audaugs.OneOf(
        [audaugs.Speed(factor=3), audaugs.TimeStretch(rate=0.25)]
    ),
    audaugs.OneOf([
        audaugs.InvertChannels(), audaugs.ChangeVolume(volume_db=6.0)
    ])
])




def apply_augmentation(signal, sr):
    audio_array = TRANSFORMS(signal, sample_rate=sr)
    return audio_array


if __name__ == "__main__":
    # aug_audio, sample_rate = audaugs.add_background_noise(audio_path, output_path=output_path)
    # audio_signal, sr = librosa.load(audio_path)
    # audio_array = TRANSFORMS(audio_signal, sample_rate=sr)
    # print(audio_array)
    # sf.write(output_path,audio_array[0],sr)
    aug_dataset(data_folder, save_folder)
    #add_reverb_dataset(data_folder, save_folder_c)

# ------ Reference list of Augmentations -------- #
# add_background_noise,
# apply_lambda,
# change_volume,
# clicks,
# clip,
# harmonic,
# high_pass_filter,
# insert_in_background,
# invert_channels,
# loop,
# low_pass_filter,
# normalize,
# peaking_equalizer,
# percussive,
# pitch_shift,
# reverb,
# speed,
# tempo,
# time_stretch,
# to_mono,
