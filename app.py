from flask import Flask, request
from flask_restful import Resource, Api
import joblib
import numpy as np

app = Flask(__name__)
api = Api(app)

class JobPrediction(Resource):
    def __init__(self):
        super(JobPrediction, self).__init__()
        self.model = joblib.load('tokenization.joblib')
        self.vectorizer = joblib.load('skills.joblib')
        self.le_experience = joblib.load('experience.joblib')
        self.le_platform = joblib.load('platform.joblib')

    def tokenize_skills(self, skills):
        return self.vectorizer.transform([skills]).toarray()

    def preprocess_data(self, data):
        experience = data.get('experience')
        skills = data.get('skills')
        encoded_experience = self.le_experience.transform([experience])
        tokenized_skills = self.tokenize_skills(skills)
        return encoded_experience, tokenized_skills

    def predict_platform(self, experience, skills):
        experience_encoded = self.le_experience.transform([experience])
        skills_tokenized = self.tokenize_skills(skills)
        user_features = np.hstack((skills_tokenized, experience_encoded.reshape(1, -1)))
        prediction = self.model.predict(user_features)
        predicted_platform = self.le_platform.inverse_transform(prediction)[0]
        return predicted_platform

    def post(self):
        data = request.get_json(force=True)
        experience = data.get('experience')
        skills = data.get('skills')

        if not isinstance(experience, str):
            return {'error': 'Experience must be a single value'}, 400

        predicted_platform = self.predict_platform(experience, skills)
        return predicted_platform

api.add_resource(JobPrediction, '/predict-job')

if __name__ == '__main__':
    app.run(debug=True)
