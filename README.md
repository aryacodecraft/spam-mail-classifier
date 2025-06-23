# Spam Mail Classifier

A machine learning-based spam email classification system using Naive Bayes algorithm.

## Description

This project implements a spam email classifier using the Multinomial Naive Bayes algorithm. It processes email text data, cleans it, and classifies messages as either spam or normal.

## Features

- Text preprocessing and cleaning
- Vectorization using CountVectorizer
- Multinomial Naive Bayes classification
- Model accuracy evaluation

## Requirements

- Python 3.8+
- pandas
- scikit-learn
- numpy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/spam-mail-classifier.git
cd spam-mail-classifier
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your dataset (`spam.csv`) in the project directory
2. Run the classifier:
```bash
python src/spam_classifier.py
```

## Dataset Format
The dataset used in this project was sourced from Kaggle:
SMS Spam Collection Dataset

The input CSV should have the following columns:
- v1: Label column (spam/normal)
- v2: Email text content

## Code Structure

- `src/spam_classifier.py`: Main classification script
- Text cleaning using regex
- Feature extraction using CountVectorizer
- Model training and prediction
- Accuracy evaluation

## Sample Output

```
Accuracy: 0.98 (or similar based on the dataset)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.