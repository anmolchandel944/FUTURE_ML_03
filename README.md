# Resume Screening System using Machine Learning

##  Project Overview

This project builds an **AI-powered Resume Screening System** that automatically analyzes resumes and predicts the most suitable **job position category** based on the candidate’s skills, experience, and career objective.

The system uses **Natural Language Processing (NLP)** and **Machine Learning** techniques to process resume text and classify it into relevant job roles.

This can help recruiters and organizations automate the initial screening process and reduce manual effort.

---

##  Objective

The goal of this project is to:

* Automatically analyze resume content
* Predict suitable job positions for candidates
* Demonstrate the use of **Machine Learning and NLP in HR technology**
* Build a simple AI-based resume classification system

---

##  Dataset

Dataset used: **Resume Dataset (Kaggle)**

Important columns used in this project:

| Column            | Description                               |
| ----------------- | ----------------------------------------- |
| career_objective  | Candidate career summary                  |
| skills            | List of technical and professional skills |
| positions         | Previous job roles                        |
| responsibilities  | Job responsibilities                      |
| job_position_name | Target job category                       |

The resume information from multiple columns is **combined into a single text feature** for machine learning processing.

---

##  Machine Learning Pipeline

The following steps are used in this project:

1. Load the dataset
2. Clean column names and handle missing values
3. Combine resume-related fields into a single text feature
4. Apply text preprocessing (lowercase, punctuation removal, stopword removal)
5. Convert text data into numerical features using **TF-IDF Vectorization**
6. Split dataset into training and testing sets
7. Train a **Naive Bayes classifier**
8. Evaluate the model using accuracy and classification report
9. Save the trained model for future predictions

---

##  Technologies Used

* Python
* Pandas
* Scikit-learn
* NLTK
* TF-IDF Vectorization
* Naive Bayes Algorithm

---

##  Project Structure

```
FUTURE_ML_03
│
├── data
│   └── resume_dataset.csv
│
├── src
│   └── resume_classifier.py
│
├── model
│   ├── resume_model.pkl
│   └── vectorizer.pkl
│
├── requirements.txt
│
└── README.md
```

---

##  Installation

Install required libraries using:

```
pip install pandas scikit-learn nltk
```

---

##  How to Run

Run the following command in the terminal:

```
python src/resume_classifier.py
```

---

##  Example Output

Example prediction:

```
Sample Resume:
Python machine learning data science spark cloud analytics big data

Predicted Job Role:
Machine Learning Engineer
```

---

##  Applications

This system can be used in:

* Recruitment automation
* HR resume filtering systems
* Job recommendation platforms
* Talent screening tools

---

##  Future Improvements

* Deploy the model as a **web application**
* Add deep learning models like **BERT or Transformers**
* Build a **resume upload interface**
* Integrate with real HR recruitment systems

---

##  Author

Anmol Chandel
Machine Learning Intern

---

##  Internship Task

This project was developed as part of the **Machine Learning Internship Program at Future Interns**.

Task: **AI Resume Screening System**
