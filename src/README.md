# Speech Emotion Recognition using keras, tensorflow and sklearn

This repository contains the code for the speech emotion recognition model built using keras, tensorflow and sklearn libraries. The data used for trainibg purposes is the RAVDESS Audio Dataset.

Tested on python 3.8.10

### Installation
To install the dependencies run:
```
pip install -r requirements.txt
```

### Ryerson Audio-Visual Database of Emotional Speech and Song 
The  RAVDESS audio dataset consists of two lexically-matched statements are vocalized in a neutral North American accent by 24 professional actors (12 female, 12 male) in the database. Calm, happy, sad, angry, afraid, surprise, and disgust expressions can be found in speech, whereas calm, happy, sad, angry, and fearful emotions can be found in song. Each expression has two emotional intensity levels (normal and strong), as well as a neutral expression. We will only be making use of audio files in this project.

The audio files of RAVDESS is selected and placed inside the `data` folder. 


### Extracting features from audio files
To extract the data, run:
```
python utils.py
```

### Train the model
To train the model, run:
```
python model.py
```

### Use the model to make predictions
To get predictions, run:
cd src 
```
python engine.py --framework=(keras/sklearn) --infer --infer-file-path="filepath of the sample to make predictions"
```
