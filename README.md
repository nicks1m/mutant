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

Navigate to <project folder>
  1) Create a folder named datasets
  2) Create two more folders inside "datasets":
     "datasets_all" and "mel_spectrograms"
  3) Extract the dataset downloaded from the link above into "datasets_all" folder
  4) Run Preprocess.py in Pycharm or any preferred method
  5) "mel_spectrograms" folder should be populated with .npy files.
  
The audio have been pre-processed.
  
## Training -- Optional Step, as Weights have been provided
  
  1) Run train.py script
  2) Model folder would be created and weights + parameters would be saved
 
## Audio Generation/Reconstruction
  
  1) Create a folder named "samples"
  2) Create four folders inside "samples":
     "original", "generated", "latent", "spec_comparison"
  3) Run generate.py script and audio files will be saved in the respective folders above.
  
  
