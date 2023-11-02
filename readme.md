# Speech Emotion Recognition on RAVDESS using Keras, Tensorflow and Sklearn

## Business Objective

Emotions are incredibly vital in the mental existence of humans. It is a means of communicating one's point of view or emotional state to others. Humans can sense the emotional state of one another through their sensory organs. Whereas doing the same for a computer is not an easy task. Although computers can quickly comprehend content-based information, obtaining the depth underlying content is challenging, which is what speech emotion recognition aims to accomplish.

The extraction of the speaker's emotional state from his or her speech signal is known as Speech Emotion Recognition (SER). There are a few universal emotions that any intelligent system with finite processing resources can be trained to recognize or synthesize as needed. These include neutral, calm, fear, anger, happiness, sad, etc. Speech emotion recognition is being more widely used in areas such as medical, entertainment, and education.

In this project, we will build a model that will be able to recognize emotions from sound files with the help of Keras and TensorFlow libraries. We will also build a model using an MLP from the scikit-learn library.

---

## Data Description

The dataset in use is the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). It contains a total of 7356 files. Two lexically-matched statements are vocalized in a neutral North American accent by 24 professional actors (12 female, 12 male) in the database. Calm, happy, sad, angry, afraid, surprise, and disgust expressions can be found in speech, whereas calm, happy, sad, angry, and fearful emotions can be found in song. Each expression has two emotional intensity levels (normal and strong), as well as a neutral expression. All three modalities are available: audio-only (16bit, 48kHz.wav), audio-video (720p H.264, AAC 48 kHz, .mp4), and video-only (720p H.264, AAC 48 kHz, .mp4) (no sound). For this particular project, we will be making use of the audio-only files.

---

## Aim

To build an Deep neural network model that is able to correctly classify speech audio files into different emotions such as happy, sad, anger, neutral, etc.

---

## Tech Stack

- Language: `Python`
- Libraries: `Keras`, `TensorFlow`, `Librosa`, `Soundfile`, `scikit-learn`, `Pandas`, `Matplotlib`, `NumPy`, `Pickle`

---

## Approach

1. Importing the required libraries and packages
2. Open the config.ini file. (This is a configuration file which can be edited according to your dataset)
3. Read the dataset (audio files)
4. Visualize the structure of a sound file
5. Understand the features of a sound file
6. Extract features
7. Load the entire data
8. Train-Test Split
9. Model Training
   - Using Keras and TensorFlow
   - Using MLP from scikit-learn
10. Hyperparameter optimization
11. Code modularization for production

---

## Modular Code Overview

```
project_root
│
├── input
│   ├── config.ini
│   ├── Audio_Song_Actors_01-24
│   └── Audio_Speech_Actors_01-24
│
├── src
│   ├── engine.py
│   └── ml_pipeline
│       ├── utils.py
│       ├── model.py
│
├── output
│   └── keras
│
└── lib
    ├── SpeechEmotions.ipynb
```

---

## Key Concepts Explored

1. Understanding the problem statement
2. Understanding the basic architecture of a neural network
3. Understanding the structure of sound waves
4. Understanding Fourier Transform
5. Understanding the file structure of the data
6. Learning visualization of sound waves using spectrograms
7. Understanding zero-crossing
8. Using libraries like Librosa and Soundfile
9. Using libraries like Matplotlib, Pandas, NumPy, etc.
10. Importing the audio files and playing them
11. Model training using Keras and TensorFlow
12. Model training using scikit-learn
13. Using MLP model
14. Creation of config files
15. Understanding Hyperparameter optimization
16. Learning how to save models


---