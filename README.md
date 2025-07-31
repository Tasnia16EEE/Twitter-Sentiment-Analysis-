# BERT Multi-label Sentiment Analysis for Twitter

This project implements a BERT-based multi-label classification model for sentiment analysis on Twitter data, classifying tweets into four categories: Positive, Negative, Neutral, and Irrelevant.

## Overview

The model uses BERT (Bidirectional Encoder Representations from Transformers) to analyze entity-specific sentiment in tweets. It's designed to understand the context and sentiment expressed about specific entities (companies, products, etc.) mentioned in tweets.

## Dataset

The project uses the **Twitter Entity Sentiment Analysis** dataset from Kaggle, which contains:
- Tweet text
- Entity mentioned (e.g., companies, products)
- Sentiment labels (Positive, Negative, Neutral, Irrelevant)

## Model Architecture

- **Base Model**: `bert-base-uncased` from Hugging Face Transformers
- **Architecture**:
  - BERT encoder for feature extraction
  - Dropout layer (0.3) for regularization
  - Linear classification head (768 → 4 outputs)
- **Loss Function**: Binary Cross-Entropy with Logits (BCEWithLogitsLoss)
- **Optimizer**: Adam

## Key Features

- **Multi-label Classification**: Handles 4 sentiment categories
- **Entity-aware**: Incorporates entity context into sentiment analysis
- **Text Preprocessing**: Includes cleaning, lowercasing, and stopword removal
- **Balanced Dataset**: Uses stratified sampling to ensure equal representation of all classes

## Requirements

```python
torch
transformers
pandas
numpy
scikit-learn
nltk
kagglehub
```

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install torch transformers pandas numpy scikit-learn nltk kagglehub
```

3. Download NLTK stopwords:
```python
import nltk
nltk.download('stopwords')
```

## Usage

### Running the Notebook

1. Open `bert_multilabel_classification_sentiment_analysis.ipynb` in Google Colab or Jupyter Notebook
2. Ensure GPU runtime is enabled (recommended for faster training)
3. Run all cells sequentially

### Model Training

The notebook includes:
- Data loading and preprocessing
- Text cleaning (removing URLs, mentions, special characters)
- One-hot encoding of sentiment labels
- Train/validation split
- Model training with configurable hyperparameters

### Hyperparameters

```python
MAX_LEN = 256
TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 1e-05
```

### Making Predictions

```python
# Example usage
example = "This product is amazing!"
encodings = tokenizer.encode_plus(
    example,
    None,
    add_special_tokens=True,
    max_length=MAX_LEN,
    padding='max_length',
    return_token_type_ids=True,
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt'
)

model.eval()
with torch.no_grad():
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    token_type_ids = encodings['token_type_ids'].to(device)
    output = model(input_ids, attention_mask, token_type_ids)
    prediction = torch.sigmoid(output).cpu().numpy()
```

## Project Structure

```
├── bert_multilabel_classification_sentiment_analysis.ipynb  # Main notebook
└── README.md                                                # This file
```

## Model Performance

The model evaluation includes:
- Classification report with precision, recall, and F1-score for each class
- Validation loss monitoring
- Per-class performance metrics

## Data Preprocessing

The preprocessing pipeline includes:
1. Text cleaning (lowercase conversion, URL removal, mention removal)
2. Special character removal
3. Stopword removal using NLTK
4. Entity context incorporation (format: "Entity. tweet_text")

## Future Improvements

- Increase training epochs for better performance
- Implement cross-validation
- Add early stopping mechanism
- Experiment with different BERT variants (RoBERTa, DistilBERT)
- Add model checkpointing
- Implement threshold optimization for multi-label predictions

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Hugging Face Transformers library
- Twitter Entity Sentiment Analysis dataset from Kaggle
- BERT paper: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
