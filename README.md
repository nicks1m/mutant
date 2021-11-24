# VAE Sample Generator

Instructions for operating the model. As of now it is still work-in-progress and it's current feature is to reconstruct .wav files.

## Setup

First, clone the repository
```clone
git clone https://github.com/nicks1m/mutant.git
```

Create a virtual environment and install the requirements:
```setup
python3 -m venv /path/to/venv/

source /path/to/venv/bin/activate.
```
*Venv should be in project root folder

*Import all libraries that are in the scripts

For more information on VENV installation
https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/


## Download
Download the Dataset
```download
https://drive.google.com/file/d/1nAXnKDASp1wotNj6RQVwHZhmOrIMxdq8/view?usp=sharing
```

## Pre-processing

Navigate to /project_folder/
  1) Create a folder named datasets
  2) Create two more folders inside "datasets":
     "datasets_all" and "mel_spectrograms"
  3) Extract the dataset downloaded from the link above into "datasets_all" folder
  4) Run Preprocess.py in Pycharm or any preferred method
  5) "mel_spectrograms" folder should be populated with .npy files.
  
The audio have been pre-processed.
  
## Training -- Unnecessary Step, as Weights have been provided
  
  1) Run train.py script
  2) Model folder would be created and weights + parameters would be saved
 
## Audio Generation/Reconstruction
  
  1) Create a folder named "samples"
  2) Create four folders inside "samples":
     "original", "generated", "latent", "spec_comparison"
  3) Run generate.py script and audio files will be saved in the respective folders above.
  
  
## Outputs

 in /samples/generated:
 
 audio reconstructions of mel-spectrograms that have been predicted by the model.
 
 in /samples/original:
 
 audio reconstructions of mel-spectrograms that have been converted from raw audio. These are not fed into the model.
 
 in /samples/latent:
 
 audio reconstructions of mel-spectrograms formed from latent vector sampling. (w.i.p)
 
 in /samples/spec_comparison:
 
 spectogram side by side comparison between original and generated
 
 
 
## Additional Helper Functions

### Data Augmentation

Due to the small dataset of samples available, I have looked into expanding the dataset via data augmentation, using Facebook Research AugLy library. I paired a couple of augmentation features together with a either/or probability that one will get chosen. For example, one of the two features in the array [(pitch_shift,harmonics)] will be chosen and applied to the audio signal. These features can be manipulated and have their intensity changed. I ran the entire folder of audio files through two transformation functions, hence we get a varied outcome. With an initial data set of 400 samples, I was able to obtain triple the size, leading to a current dataset of 1200. This allows the model to train and generalize better.

### Mel-Spec Plotter

I added a visualize_spec script that is able to take a single or double audio signal and plot it using librosa.specshow. This plots the two signals above one another, and we are able to look at the mel-scale frequency domain of the respective signals, to get a rough understanding of the signal composition. This is automatically generated everytime we generate a new set of samples from the model.

## Additional Notes

The training of the model was done in Google Colab, utilizing their backend GPU due to insufficient processing power on my laptop. I can provide a link to the Colab notebook if you would like to train your a large dataset and have a go. All that is required is the processed spectrograms from preprocessing.py. The weights and parameters will be saved onto the Colab project folder and it can be downloaded onto the local project folder for generation.
