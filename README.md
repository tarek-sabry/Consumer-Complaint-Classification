# 📩 Consumer Complaint Classification Using Deep Learning

## 📌 Project Overview

This project aims to automatically classify consumer complaints into five product categories using Natural Language Processing (NLP) and Deep Learning techniques.

The project compares multiple deep learning architectures, including SimpleRNN, LSTM, GRU, and DistilBERT, to identify the best-performing model for complaint classification.

A web application was also developed using Streamlit to allow users to enter complaints and receive instant predictions.

## 🎯 Project Objectives

- Analyze and preprocess consumer complaint text data.
- Build and train multiple NLP classification models.
- Compare SimpleRNN, LSTM, GRU, and DistilBERT.
- Evaluate model performance using Accuracy, Precision, Recall, and F1-Score.
- Select the best-performing model.
- Deploy the final model using Streamlit.

## 📂 Dataset

**Source:** [Consumer Complaints Dataset – Kaggle](https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp)

The dataset contains consumer complaints and their corresponding product categories.

**Main Features:**
- `narrative`: Consumer complaint text.
- `product`: Target classification category.

**Target Classes:**
1. credit_card
2. credit_reporting
3. debt_collection
4. mortgages_and_loans
5. retail_banking

## ⚙️ Project Workflow

### 1. Data Loading & Exploration
- Loaded the dataset.
- Explored data structure and target classes.
- Identified missing values and duplicate records.

### 2. Data Cleaning & Preprocessing
- Removed missing values.
- Removed duplicate records.
- Converted text to lowercase.
- Removed punctuation and special characters.
- Removed stopwords while preserving important negation words.
- Applied lemmatization.

### 3. Text Vectorization
- Tokenization using Keras Tokenizer.
- Sequence conversion.
- Padding and truncation.
- Maximum vocabulary size: 20,000.
- Maximum sequence length: 200.

### 4. Model Development

Four models were developed and evaluated:

- SimpleRNN
- LSTM
- GRU
- DistilBERT (Transformer)

### 5. Model Evaluation

Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

## 📊 Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| **GRU** | **84.86%** | **84.88%** | **84.86%** | **84.87%** |
| DistilBERT | 79.78% | 80.02% | 79.78% | 79.80% |
| LSTM | 60.24% | 44.32% | 60.24% | 50.97% |
| SimpleRNN | 45.98% | 26.83% | 45.98% | 30.64% |

### 🏆 Best Model: GRU

The GRU model achieved the best performance with:

- Accuracy: **84.86%**
- Precision: **84.88%**
- Recall: **84.86%**
- F1-Score: **84.87%**

Based on the current experimental results, GRU was selected as the final classification model.

**Experimental note:** DistilBERT was fine-tuned on a smaller training subset than the recurrent models. Therefore, the comparison is not fully controlled for training sample size.

## 🧪 New Complaint Prediction

The selected GRU model was tested on previously unseen customer complaints.

**Example 1:**

Input:
> I noticed an unauthorized transaction on my credit card. I contacted the bank but the issue has not been resolved.

Prediction:
- Category: `credit_card`
- Confidence: **73.71%**

**Example 2:**

Input:
> I have been receiving repeated calls from a debt collector regarding an account I do not recognize.

Prediction:
- Category: `debt_collection`
- Confidence: **93.96%**

## 🌐 Streamlit Web Application

A web application was developed to make the trained model accessible through a simple user interface.

**Application Features:**
- Enter a customer complaint.
- Predict the complaint category.
- Display the prediction confidence score.
- Visualize probabilities across all five categories.
- Clear the input and perform another prediction.

### Run Locally

1. Clone the repository:

```bash
git clone https://github.com/tarek-sabry/Consumer-Complaint-Classification
```

2. Navigate to the project directory:

```bash
cd Consumer-Complaint-Classification
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
python -m streamlit run app.py
```

5. Open the local URL displayed in your terminal.

## 🛠️ Technologies & Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- PyTorch
- Hugging Face Transformers
- NLTK
- Matplotlib / Seaborn
- Streamlit

## 📁 Project Structure

```text
Consumer-Complaint-Classification/
│
├── app.py
├── best_gru_model.keras
├── tokenizer.pkl
├── label_encoder.pkl
├── config.json
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Future Improvements

- Handle class imbalance using advanced sampling techniques.
- Perform more extensive hyperparameter tuning.
- Compare models using identical training data and experimental settings.
- Improve generalization on unseen complaints.
- Deploy the application online using Streamlit Community Cloud.

## 👨‍💻 Author

**Tarek Sabry**

Data Science & Data Analytics Enthusiast

