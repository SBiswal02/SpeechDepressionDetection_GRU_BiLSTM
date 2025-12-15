# Speech Depression Detection using GRU and BiLSTM

A multimodal deep learning approach for depression detection using speech, text, and prosodic features from the DAIC-WoZ corpus. This project implements several neural network architectures including GRU (Gated Recurrent Unit) and Bidirectional LSTM (BiLSTM) for binary classification of depression.

## Overview

This project implements a comprehensive depression detection system that leverages multiple modalities:
- **Text**: Transcript-based analysis using BiLSTM
- **Speech**: Audio feature-based analysis using GRU
- **Prosody**: Prosodic features (pitch, jitter, shimmer, pauses, speech rate)
- **Multimodal Fusion**: Combining text and speech, or text, speech, and prosody

## Dataset

This project uses the **DAIC-WoZ (Distress Analysis Interview Corpus - Wizard of Oz)** dataset. The dataset contains:
- Audio recordings of clinical interviews
- Transcripts of the interviews
- PHQ-8 (Patient Health Questionnaire) scores
- Binary depression labels (PHQ_Binary: 0 = not depressed, 1 = depressed)

### Dataset Structure
- `Participant_ID`: Unique identifier for each participant
- `Gender`: Participant gender
- `PHQ_Binary`: Binary depression label (0 or 1)
- `PHQ_Score`: Continuous PHQ-8 score
- `Audio_Path`: Path to audio files (format: `{Participant_ID}_AUDIO.wav`)
- `Transcript`: Text transcriptions (format: `{Participant_ID}_TRANSCRIPT.csv`)
- `Features`: Extracted audio features

### Data Folder Structure

The project includes a `data/` folder containing the dataset organization and preprocessing scripts:

```
data/
├── extract_audios.ipynb              # Data preprocessing notebook
│                                    # Functions to extract audio files and transcripts
│                                    # from DAIC-WoZ dataset structure
│
└── labels/                           # Label files and data splits
    ├── Detailed_PHQ8_Labels.csv     # Detailed PHQ-8 questionnaire responses
    │                                # Contains: Participant_ID, PHQ_8NoInterest,
    │                                # PHQ_8Depressed, PHQ_8Sleep, PHQ_8Tired,
    │                                # PHQ_8Appetite, PHQ_8Failure, 
    │                                # PHQ_8Concentrating, PHQ_8Moving, PHQ_8Total
    │
    ├── train_split.csv              # Training set labels
    │                                # Contains: Participant_ID, Gender, PHQ_Binary,
    │                                # PHQ_Score, PCL-C (PTSD), PTSD Severity
    │
    ├── test_split.csv               # Test set labels (AVEC2017 format)
    ├── dev_split.csv                # Development/validation set labels
    │
    └── edited/                      # Edited/preprocessed label files
        ├── Detailed_PHQ8_Labels.csv
        ├── train_split.csv
        └── dev_split.csv
```

### Data Preprocessing

The `data/extract_audios.ipynb` notebook contains utilities for organizing DAIC-WoZ data:

- **`copy_audios()`**: Extracts audio files (`{ID}_AUDIO.wav`) from participant directories
- **`copy_transcripts()`**: Extracts transcript files (`{ID}_Transcript.csv`) 
- **`copy_transcripts_allcaps()`**: Extracts transcript files with uppercase naming (`{ID}_TRANSCRIPT.csv`)

The preprocessing handles the DAIC-WoZ directory structure where each participant has a folder named `{Participant_ID}_P/` containing:
- Audio files: `{Participant_ID}_AUDIO.wav`
- Transcript files: `{Participant_ID}_TRANSCRIPT.csv` or `{Participant_ID}_Transcript.csv`

### Data Splits

The project uses predefined train/test/validation splits:
- **Training Set**: ~148-244 samples (varies by modality)
- **Test/Dev Set**: ~48-62 samples
- Splits are available in both standard and AVEC2017 challenge formats

## Project Structure

```
SpeechDepressionDetection/
├── README.md
├── data/                      # Dataset organization and preprocessing
│   ├── extract_audios.ipynb  # Data extraction utilities
│   └── labels/               # Label files and data splits
│       ├── Detailed_PHQ8_Labels.csv
│       ├── train_split.csv
│       ├── test_split.csv
│       ├── dev_split.csv
│       └── edited/           # Preprocessed label files
│
├── TextOnly/                  # Text-based depression detection
│   ├── main.ipynb            # Main notebook for text classification
│   ├── bilstm_*.h5           # Trained BiLSTM models
│   └── README.md             # TextOnly module documentation
│
├── SpeechOnly/               # Audio-based depression detection
│   ├── main.ipynb            # Main preprocessing notebook
│   ├── gru_*.ipynb           # GRU model variants
│   ├── audio_*.csv           # Processed audio feature datasets
│   └── trial.py              # Trial implementation with regularization
│
├── ProsodyOnly/              # Prosodic features extraction
│   ├── main.ipynb            # Main prosody analysis notebook
│   ├── prosodic_feat.ipynb   # Feature extraction
│   └── *.csv                 # Extracted prosodic features
│
├── TextSpeech/               # Multimodal fusion (Text + Speech)
│   ├── main.ipynb            # Fusion model implementation
│   └── fusion_model.h5       # Trained fusion model
│
└── TextSpeechProsody/        # Multimodal fusion (Text + Speech + Prosody)
    ├── main.ipynb            # Complete fusion model
    └── model.h5              # Trained multimodal model
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

## Installation

### Requirements

```bash
# Core libraries
pip install numpy pandas seaborn matplotlib

# Machine Learning
pip install scikit-learn
pip install tensorflow tensorflow-datasets tensorflow-hub

# Deep Learning (PyTorch - optional)
pip install torch torchvision torchaudio

# Audio processing
pip install librosa python-speech-features soundfile
pip install pyAudioAnalysis parselmouth

# NLP
pip install -U spacy nltk
python -m spacy download en_core_web_sm

# Additional utilities
pip install wordcloud scipy
```

### Complete Installation Script

```bash
pip install numpy pandas seaborn matplotlib wordcloud scipy
pip install -U scikit-learn
pip install tensorflow tensorflow-datasets tensorflow-hub
pip install librosa python-speech-features soundfile pyAudioAnalysis parselmouth
pip install -U spacy nltk
python -m spacy download en_core_web_sm
```

## Usage

### Step 1: Data Preparation

First, prepare your DAIC-WoZ dataset:

```python
# Navigate to data directory
cd data

# Run extract_audios.ipynb to organize the dataset
# This will extract audio files and transcripts from the DAIC-WoZ structure
# Update the source_directory and destination_folder paths in the notebook
```

### Step 2: Choose Your Model

#### 1. Text-Only Classification

```python
# Navigate to TextOnly directory
cd TextOnly

# Open and run main.ipynb
# The notebook loads labels from data/labels/edited/
# Trains a BiLSTM model on text transcripts
```

#### 2. Speech-Only Classification

```python
# Navigate to SpeechOnly directory
cd SpeechOnly

# Preprocess audio data
# Run main_prep.ipynb for data preparation and feature extraction
# Run gru_*.ipynb notebooks for model training
# Choose from different GRU architectures (32_16, 64_32, etc.)
```

#### 3. Multimodal Fusion

```python
# Navigate to TextSpeech or TextSpeechProsody directory
cd TextSpeech  # or TextSpeechProsody

# Open main.ipynb
# The notebook loads:
#   - Labels from data/labels/edited/
#   - Text features from transcripts
#   - Audio features (pre-extracted)
#   - Prosodic features (for TextSpeechProsody)
# Train the fusion model
```

## Training

The models use the following training configurations:
- **Loss Function**: Categorical cross-entropy (binary classification)
- **Optimizer**: Adam
- **Metrics**: Accuracy
- **Early Stopping**: Implemented to prevent overfitting
- **Validation Split**: Typically 20% of training data

### Data Preprocessing

#### 1. Dataset Organization
First, organize the raw DAIC-WoZ dataset using `data/extract_audios.ipynb`:
- Extract audio files from participant directories
- Extract transcript files
- Organize into train/validation/test directories

#### 2. Feature Extraction
Extract features from organized data:
- **Text**: Tokenization and padding from transcript files
- **Audio**: Feature extraction using librosa/python_speech_features
- **Prosody**: Pitch, jitter, shimmer, pause analysis using parselmouth

#### 3. Data Splits
The project uses predefined splits from `data/labels/`:
- Load train/test splits from CSV files
- Use `train_split.csv` for training
- Use `dev_split.csv` or `test_split.csv` for evaluation
- Splits are based on Participant_ID matching

#### 4. Data Balancing
- Handle class imbalance using downsampling or upsampling
- Multiple balanced datasets available (e.g., `audio_balanced.csv`, `audio_balanced_downsampled.csv`)

## Evaluation Metrics

The models are evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

## Results

Model performance varies across different modalities and architectures. The multimodal fusion approaches (Text+Speech, Text+Speech+Prosody) generally show improved performance compared to unimodal approaches.

## Notes

- The DAIC-WoZ dataset requires proper access and licensing
- Audio features are extracted and preprocessed before training
- Text data is tokenized and padded to fixed-length sequences
- Models support binary classification (depressed/not depressed) based on PHQ-8 scores

## Citation

If you use this code, please cite the DAIC-WoZ dataset:

```
Gratch, J., Artstein, R., Lucas, G. M., Stratou, G., Scherer, S., Nazarian, A., ... 
& Rizzo, A. (2014). The Distress Analysis Interview Corpus of human and computer 
interviews. In LREC (pp. 3123-3128).
```

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

[Your contact information]
