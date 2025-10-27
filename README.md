# 🧠 Detecting Political Affiliation and Linguistic Patterns in Reddit Discourse

**Author:** Faizan Waheed  
**Email:** fawaheed@iu.edu  
**Institution:** Indiana University Bloomington  
**Date:** April 2025  

---

## 📘 Overview

This project explores how **political ideology is expressed through language** on Reddit.  
By analyzing posts from two ideologically distinct subreddits — **r/Democrats** and **r/Republicans** —  
we aim to detect political affiliation and uncover linguistic differences using both **traditional machine learning models** and **state-of-the-art transformer architectures**.

The study combines **classification**, **linguistic feature analysis**, and **visualization techniques** to better understand how online communities linguistically convey ideology.

The full research paper is available as:  
📄 **Project Report.pdf**

---

## 🎯 Research Objectives

- Classify Reddit posts by political affiliation using textual data.  
- Compare the performance of **classical ML models** and **transformer-based NLP models**.  
- Identify linguistic, lexical, and sentiment differences between partisan groups.  
- Visualize thematic patterns using **word clouds** and **BERTopic** for interpretability.  

---

## 📊 Dataset

**Data Source:** Reddit API (via PRAW – Python Reddit API Wrapper)  
**Communities:** `r/Democrats` and `r/Republicans`  
**Total Posts:** 1,989 (997 Democrat | 992 Republican)

### Data Fields
- `text`: Combined title and body of Reddit posts.  
- `label`: Binary tag representing the subreddit (Democrat or Republican).  

**Data Ethics:**  
- Only publicly available Reddit content was used.  
- No user-identifiable information was collected or stored.  

---

## ⚙️ Methodology

### 🔄 Preprocessing
- Lowercasing, punctuation & URL removal  
- Stopword removal using **NLTK**  
- Lemmatization using **WordNetLemmatizer**  
- TF-IDF vectorization (for classical models)  
- Tokenization and truncation (for transformer models)

### 🧩 Models Implemented
| Model Type | Algorithms Used | Description |
|-------------|----------------|--------------|
| Classical ML | Logistic Regression, SVM, Random Forest | Feature-based classifiers using TF-IDF |
| Transformer-based | DistilBERT, BART, Twitter RoBERTa | Fine-tuned deep contextual models using HuggingFace Transformers |

### 🧠 Training Strategy
- Stratified 80/20 train-test split  
- Hyperparameter tuning using **GridSearchCV**  
- Regularization (L2, dropout, weight decay)  
- Early stopping and cosine learning rate scheduling for transformers  
- Evaluation metrics: **Accuracy**, **Precision**, **Recall**, **F1-score**

### 🔍 Linguistic Analysis
- **Lexical Richness**
- **Pronoun Usage**
- **Readability (Flesch Reading Ease)**
- **Sentiment Analysis** using VADER  
- **Word Clouds** to visualize top terms across subreddits

---

## 🧪 Key Results

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|-----------|----------|-----------|
| Logistic Regression | 0.623 | 0.582 | **0.885** | **0.702** |
| SVM | 0.648 | 0.613 | 0.773 | 0.689 |
| Random Forest | 0.633 | 0.598 | **0.860** | **0.702** |
| DistilBERT | **0.661** | 0.647 | 0.655 | 0.679 |
| BART | 0.646 | **0.681** | 0.555 | 0.612 |
| Twitter RoBERTa | 0.651 | 0.645 | 0.680 | 0.662 |

**Highlights:**
- Transformer models outperformed classical ones in precision and interpretive depth.  
- Classical models achieved **higher recall and balanced F1-scores**, showing strong generalization on medium-sized data.  
- **DistilBERT** achieved the best overall accuracy.  

---

## 💬 Linguistic Insights

| Subreddit | Lexical Richness | Pronoun Ratio | Readability | Sentiment |
|------------|------------------|----------------|--------------|------------|
| r/Democrats | 0.7823 | **0.0815** | 65.14 | **0.0301 (positive)** |
| r/Republicans | **0.8062** | 0.0708 | **66.40** | 0.0021 (neutral) |

**Interpretation:**
- Democrats use **more personal and emotional language**.  
- Republicans display **higher vocabulary diversity and readability**.  
- Sentiment analysis shows slightly more positive tone among Democrats.  
- Word clouds confirm shared political focus but distinct framing and emotional tone.

---

## 🧮 Tools & Libraries
- `Python 3.10+`
- `pandas`, `numpy`, `matplotlib`, `scikit-learn`
- `nltk`, `spacy`, `wordcloud`, `vaderSentiment`
- `torch`, `transformers` (HuggingFace)
- `BERTopic` (optional for topic modeling)

---

## 📈 Conclusions

- Political ideology can be **reasonably inferred from Reddit language** using NLP.  
- **Classical ML models** remain competitive, particularly on small datasets.  
- **Transformer-based models** capture richer context and nuanced phrasing.  
- Linguistic patterns reveal that **engagement tone, vocabulary, and sentiment** align with partisan identity.  
- These findings bridge **computational linguistics** and **political communication**, providing insights for researchers, policymakers, and data scientists.

---

## 🧩 Repository Structure
