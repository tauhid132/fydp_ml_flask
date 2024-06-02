import joblib
import numpy as np

model = joblib.load('tokenization.joblib')
vectorizer = joblib.load('skills.joblib')
le_experience = joblib.load('experience.joblib')
le_platform = joblib.load('platform.joblib')

def predict_platform(experience, skills):
    experience_encoded = le_experience.transform([experience])
    skills_tokenized = vectorizer.transform([skills]).toarray()
    user_features = np.hstack((skills_tokenized, experience_encoded.reshape(1, -1)))
    predicted_platform = model.predict(user_features)
    predicted_platform = le_platform.inverse_transform(predicted_platform)
    
    return predicted_platform[0]

user_experience = 'Intermediate'
user_skills = 'UI/UX Design'
predicted_platform = predict_platform(user_experience, user_skills)
print("Predicted platform:", predicted_platform)
