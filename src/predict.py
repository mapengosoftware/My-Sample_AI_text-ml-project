import joblib

# Load model
model = joblib.load('model.pkl')

# Take user input
text = input("Enter a movie review: ")
prediction = model.predict([text])
print("Prediction:", prediction[0])
