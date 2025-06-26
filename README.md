# Document Classification using TF-IDF and Word2Vec

This project focuses on classifying documents such as invoices, e-way bills, delivery challans, etc., using natural language processing (NLP) and machine learning models. Two approaches are implemented:

- TF-IDF with machine learning models
- Word2Vec embeddings with machine learning models

---

## Dataset

- `results.csv` – contains labeled document text data
- `Demo_2.csv` – validation dataset for final model testing

Each file includes:
- `text`: the content of the document
- `document`: the label or type of the document

---

## Steps Involved

1. **Data Loading**
   - Load CSV files using pandas

2. **Text Preprocessing**
   - Convert text to lowercase
   - Remove punctuation
   - Tokenize the text
   - Remove stopwords
   - Lemmatize words
   - Filter out very short tokens (2–3 characters)

3. **Feature Extraction**
   - **TF-IDF**: Convert text into TF-IDF vectors
   - **Word2Vec**: Train Word2Vec model, average word vectors per document

4. **Model Training**
   - Linear Support Vector Classifier (SVC)
   - Random Forest Classifier
   - XGBoost Classifier

5. **Evaluation**
   - Evaluate using accuracy, precision, recall, F1-score
   - Confusion matrix for visual performance

6. **Validation**
   - Best model tested on `Demo_2.csv` (unseen data)

---

## TF-IDF

## Accuracy & Results on Result.csv

| Model           | Train Accuracy | Test Accuracy |
|----------------|----------------|---------------|
| Linear SVC      | 99.7           | 96.4          |
| Random Forest   | 99.2           | 97.1          |
| XGBoost         | 100            | 97.5          |


**Validation Accuracy on `Demo_2.csv`**: 
XGBoost : 78%

---

## Confusion Matrix (Sample)
![image](https://github.com/user-attachments/assets/6ae5357b-4e92-4f72-aa93-593a73c8b77f)
---

## Word2Vec

## Accuracy & Results on Result.csv

| Model           | Train Accuracy | Test Accuracy |
|----------------|----------------|---------------|
| Linear SVC      | 98.4           | 97.3          |
| Random Forest   | 99.2           | 97.1          |
| XGBoost         | 99.9           | 96.7          |

---

## Confusion Matrix (Sample)
![image](https://github.com/user-attachments/assets/f316fd9e-e8c6-4d6a-8f49-7dde22fcf618)

## Requirements

Install the required Python packages:

```bash
pip install pandas nltk gensim scikit-learn seaborn xgboost numpy
````

---

## Files in This Repo

* `Document_classifier_tfidf.ipynb` – TF-IDF based classification
* `Document_classifier_w2v.ipynb` – Word2Vec based classification
* `README.md` – Project description and instructions

---

## Notes

* You can improve performance by tuning model hyperparameters or experimenting with different vectorization techniques.
* Best results so far were achieved using XGBoost on TF-IDF vectors.

## Next Steps
* Improve the quality and balance of training data
* Apply hyperparameter tuning (e.g., GridSearchCV, Optuna) to boost performance
* Focus on improving model performance on unseen/validation data
* Pipeline the full workflow from preprocessing to prediction
* Save the best model using joblib or pickle for inference and deployment
