import pandas as pd
import nltk
import string
import pickle

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Load dataset
data = pd.read_csv("data/resume_dataset.csv", encoding="utf-8")

print("Dataset Loaded")
print(data.head())

# Clean column names (removes hidden characters like BOM)
data.columns = data.columns.str.strip().str.replace('\ufeff','')

print("\nColumns detected:")
print(data.columns)

# Rename problematic column if needed
if 'job_position_name' not in data.columns:
    for col in data.columns:
        if 'job_position' in col:
            data.rename(columns={col:'job_position_name'}, inplace=True)

# Fill missing values
data = data.fillna("")

# Combine resume related fields
data['resume_text'] = (
    data['career_objective'].astype(str) + " " +
    data['skills'].astype(str) + " " +
    data['positions'].astype(str) + " " +
    data['responsibilities'].astype(str)
)

# Target label
y = data['job_position_name']
X = data['resume_text']

# Text cleaning function
def clean_text(text):

    text = text.lower()

    text = text.translate(str.maketrans('', '', string.punctuation))

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

print("\nCleaning resume text...")

X_clean = X.apply(clean_text)

# Convert text to numbers
vectorizer = TfidfVectorizer(max_features=3000)

X_vectorized = vectorizer.fit_transform(X_clean)

print("Vectorization complete")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()

model.fit(X_train, y_train)

print("\nModel trained successfully")

# Predictions
predictions = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

print("\nClassification Report")
print(classification_report(y_test, predictions))

# Save model
pickle.dump(model, open("model/resume_model.pkl","wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl","wb"))

print("\nModel and vectorizer saved successfully")

# Test prediction
sample_resume = [
    "Python machine learning data science spark cloud analytics big data"
]

sample_vector = vectorizer.transform(sample_resume)

prediction = model.predict(sample_vector)

print("\nSample Resume:", sample_resume[0])
print("Predicted Job Role:", prediction[0])