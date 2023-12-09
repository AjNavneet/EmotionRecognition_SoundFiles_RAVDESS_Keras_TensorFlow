# Speech Emotion Recognition on RAVDESS using Keras, TensorFlow, and Scikit-Learn

## Business Objective

Understanding human emotions is crucial for effective communication. While humans effortlessly interpret emotions through speech, teaching computers to do the same is a challenging task. Speech Emotion Recognition (SER) aims to extract emotional states from speech signals, finding applications in medical, entertainment, and educational fields.

In this project, we will build models using Keras, TensorFlow, and Scikit-Learn to recognize emotions from sound files, focusing on the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).

---

## Data Description

The RAVDESS dataset contains 7356 audio files with lexically-matched statements spoken by 24 professional actors. It includes emotions such as calm, happy, sad, angry, afraid, surprise, and disgust, each with two intensity levels (normal and strong). We will use the audio-only files for this project.

---

## Aim

Develop a deep neural network model to accurately classify speech audio files into different emotions, such as happy, sad, anger, neutral, etc.

---

## Tech Stack

- **Language**: `Python`
- **Libraries**: `Keras`, `TensorFlow`, `Librosa`, `Soundfile`, `scikit-learn`, `Pandas`, `Matplotlib`, `NumPy`, `Pickle`

---

## Approach

1. **Importing Required Libraries and Packages**
2. **Configuration File**
   - Open the `config.ini` file for dataset configuration.
3. **Read Dataset (Audio Files)**
4. **Visualize Structure of a Sound File**
5. **Understand Features of a Sound File**
6. **Extract Features**
7. **Load Entire Data**
8. **Train-Test Split**
9. **Model Training**
   - Using Keras and TensorFlow
   - Using MLP from Scikit-Learn
10. **Hyperparameter Optimization**
11. **Code Modularization for Production**

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
