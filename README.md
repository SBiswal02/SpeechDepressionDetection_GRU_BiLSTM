# Speech Depression Detection using GRU and BiLSTM

A multimodal deep learning approach for depression detection using speech, text, and prosodic features. This project implements several neural network architectures including GRU (Gated Recurrent Unit) and Bidirectional LSTM (BiLSTM) to classify if the user shows signs of depression or not.

## Overview

This project implements a comprehensive depression detection system that leverages multiple modalities:
- **Dataset**: DAIC-WoZ Dataset (Distress Analysis Interview Corpus – Wizard-of-Oz) widely used for mental-health analysis.
- **Text**: Transcript-based analysis using BiLSTM.
- **Speech**: Audio feature-based analysis using GRU.
- **Prosody**: Prosodic features (pitch, jitter, shimmer, pauses, speech rate) used to analyse speech patterns
- **Multimodal Fusion**: Combining the above modalities to make the comprehensive machine learning model to detect depression

## Dataset

This project uses the **DAIC-WoZ (Distress Analysis Interview Corpus - Wizard of Oz)** dataset. The dataset contains:
- Audio recordings of clinical interviews and transcripts of the interviews
- Collected as part of the USC Institute for Creative Technologies (ICT) project
- Designed to support automatic analysis of psychological distress
- Interviews are conducted with a virtual agent (“Ellie”), controlled by a human interviewer behind the scenes (Wizard-of-Oz setup)
- Around 189 subjects - Each participant completed a semi-structured clinical interview
- Ground-truth labels:
> PHQ-8 depression score (0–24)
> Binary depression label PHQ Binary

### Dataset Structure
- `Participant_ID`: Unique identifier for each participant
- `Gender`: Participant gender
- `PHQ_Binary`: Binary depression label (0 or 1)
- `PHQ_Score`: Continuous PHQ-8 score
- `Audio_Path`: Path to audio files (format: `{Participant_ID}_AUDIO.wav`)
- `Transcript`: Text transcriptions (format: `{Participant_ID}_TRANSCRIPT.csv`)
- `Features`: Extracted audio features

### Data Splits

The project uses predefined train/test/validation splits:
- **Training Set**: ~148-244 samples (varies by modality)
- **Test/Dev Set**: ~48-62 samples
- Splits are available in both standard and AVEC2017 challenge formats

## Project Structure

```
SpeechDepressionDetection/
├── README.md
│
├── data/                     
│   ├── extract_audios.ipynb  
│   └── labels/               
│       ├── Detailed_PHQ8_Labels.csv
│       ├── train_split_Depression_AVEC2017.csv
│       ├── test_split_Depression_AVEC2017.csv
│       └── edited/           
│           ├── Detailed_PHQ8_Labels.csv
│           ├── train_split.csv
│           └── dev_split.csv
│
├── src/                      
│   ├── TextOnly/             
│   │   ├── main.ipynb        
│   │   ├── init.ipynb        
│   │   └── README.md        
│   │
│   ├── SpeechOnly/           
│   │   ├── main.ipynb        
│   │   ├── main_prep.ipynb   
│   │   ├── main_fear.ipynb   
│   │   ├── gru_32_16_1256.ipynb   
│   │   ├── gru_32_16 _1616.ipynb
│   │   ├── gru_64_32 _1616.ipynb
│   │   ├── gru_downsample.ipynb
│   │   ├── trial.py          
│   │   └── loupe_keras.py
│   │
│   ├── ProsodyOnly/          
│   │   ├── main.ipynb       
│   │   ├── prosodic_feat.ipynb        
│   │   └── check.ipynb 
│   │
│   ├── TextSpeech/         
│   │   └── main.ipynb        
│   │
│   └── TextSpeechProsody/   
│       └── main.ipynb       
│
└── results/                  
    ├── fusion_model.h5       # TextSpeech fusion model
    ├── model.h5             
    ├── TextOnly/          
    │   ├── bilstm_10.h5
    │   ├── bilstm_50.h5
    │   ├── bilstm_100.h5
    │   ├── bilstm_1k.h5
    │
    ├── SpeechOnly/        
    │   ├── bilstm_10_up.h5
    │   ├── bilstm_50_up.h5
    │   └── bilstm_100_up.h5
    │
    └── ProsodyOnly/    
        ├── pitch_contour.png
        └── pitch_contour2.png
```

## Features

### Text Features
- Tokenized and padded text sequences
- Word embeddings
- Processed transcripts from DAIC-WoZ interviews

### Audio Features
- MFCC (Mel-frequency Cepstral Coefficients)
- Spectral features
- Reshaped to (16, 16) for GRU input

### Prosodic Features
- Pitch contour
- Jitter and shimmer (voice quality measures)
- Pause patterns
- Speech rate
- Windowed prosodic features

## Models

### 1. Text-Only Model (BiLSTM)
- **Architecture**: Embedding → Bidirectional LSTM → Dropout → Dense
- **Input**: Text sequences (padded)
- **Output**: Binary classification (depressed/not depressed)

### 2. Speech-Only Model (GRU)
- **Architecture**: GRU → Dropout → Dense → Dropout → Dense
- **Input**: Audio features (16 × 16)
- **Output**: Binary classification
- **Variants**: Different GRU units (32, 64) and hidden layers

### 3. Text-Speech Fusion Model
- **Text Branch**: Embedding → Bidirectional LSTM → Dropout
- **Audio Branch**: GRU → Dropout → Dense → Dropout
- **Fusion**: Concatenate → Dense → Dropout → Output
- **Input**: Text sequences + Audio features
- **Output**: Binary classification

### 4. Text-Speech-Prosody Fusion Model
- **Text Branch**: Embedding → Bidirectional LSTM → Dropout
- **Audio Branch**: GRU → Dropout → Dense → Dropout
- **Prosody Branch**: GRU with Masking → Dropout
- **Fusion**: Concatenate all branches → Dense layers → Output
- **Input**: Text sequences + Audio features + Prosodic features
- **Output**: Binary classification

## Requirements

```
numpy
pandas
seaborn
matplotlib
scikit-learn
tensorflow
tensorflow-datasets
tensorflow-hub
torch
torchvision
torchaudio
librosa
python-speech-features
soundfile
pyAudioAnalysis
parselmouth
spacy
nltk
wordcloud
scipy
```

## Training

The models use the following training configurations:
- **Loss Function**: Categorical cross-entropy (binary classification)
- **Optimizer**: Adam
- **Metrics**: Accuracy
- **Early Stopping**: Implemented to prevent overfitting
- **Validation Split**: Typically 20% of training data


## Results

Model performance varies across different modalities and architectures. The multimodal fusion approaches (Text+Speech, Text+Speech+Prosody) generally show improved performance compared to unimodal approaches.