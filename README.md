# 🧠 Detecting Political Affiliation and Linguistic Patterns in Reddit Discourse

**Author:** Faizan Waheed  
**Email:** fawaheed@iu.edu  
**Date:** April 2025  
**Institution:** Indiana University Bloomington  

---

## 📘 Overview
This project explores how political ideology is expressed through language on Reddit by analyzing user-generated posts from **r/Democrats** and **r/Republicans**.  
Using both **traditional machine learning** and **transformer-based NLP models**, it predicts political affiliation from text and uncovers linguistic and thematic differences between the two communities.

The full research paper is included in this repository as  
📄 **“Assignment 3 Paper.pdf.”**

---

## 🧩 Objectives
- Classify Reddit posts by political affiliation using textual content only.  
- Compare the performance of classical ML models and state-of-the-art transformers.  
- Analyze linguistic and sentiment differences between partisan communities.  
- Visualize dominant words and themes to understand ideological framing.

---

## 📊 Dataset
Data was collected using the **Python Reddit API Wrapper (PRAW)** from two subreddits:
- **r/Democrats**  
- **r/Republicans**

**Sampling Method:** Top 1,000 posts per subreddit using Reddit’s native ranking algorithm.  
**Total Dataset Size:** 1,989 posts (Democrats: 997 | Republicans: 992)

All data was collected in compliance with Reddit’s API Terms of Service.  
No user identifiers or personal information were stored.

---

## ⚙️ Methodology

### 1. **Preprocessing**
- Text normalization (lowercasing, punctuation & URL removal)  
- Stopword removal (NLTK) and lemmatization (WordNetLemmatizer)  
- TF-IDF vectorization for classical models  
- Tokenization for transformer models (DistilBERT, BART, RoBERTa)

### 2. **Modeling**
| Model Type | Models Used | Tools/Libraries |
|-------------|--------------|----------------|
| Classical ML | Logistic Regression, SVM, Random Forest | scikit-learn |
| Transformer-based | DistilBERT, BART, Twitter-RoBERTa | HuggingFace Transformers |

Each model was tuned using stratified 5-fold cross-validation and evaluated with **Accuracy**, **Precision**, **Recall**, and **F1-score**.

### 3. **Linguistic Analysis**
- **Lexical Richness** (vocabulary diversity)  
- **Pronoun Usage** and **Readability Scores (Flesch Reading Ease)**  
- **Sentiment Analysis** using VADER  
- **Word Clouds** for thematic visualization (WordCloud library)

---

## 🧪 Key Findings
- **DistilBERT** achieved the highest accuracy (0.661) among transformers.  
- **Logistic Regression** and **Random Forest** achieved the top F1-scores (≈0.70).  
- **Democratic posts**: more personal language, slightly more positive tone.  
- **Republican posts**: higher lexical diversity, greater readability.  

Linguistic and visual analyses showed both communities focus on shared political topics (e.g., elections, leaders) but differ in tone, framing, and emotional expression.


