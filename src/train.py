import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Load data
df = pd.read_csv('data/imdb.csv')
X = df['text']
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# Train model
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, 'model.pkl')

# Evaluate
accuracy = pipeline.score(X_test, y_test)
print(f'Model accuracy: {accuracy:.2f}')