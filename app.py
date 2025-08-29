from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
import joblib
import os

app = Flask(__name__)

# Load and prepare the model
def load_model():
    heart_data = pd.read_csv('heart_disease_data.csv')
    X = heart_data.drop(columns='target', axis=1)
    Y = heart_data['target']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    nb_model = GaussianNB()
    lr_model = LogisticRegression()
    meta_model = LogisticRegression()
    stack_model = StackingClassifier(
        estimators=[('nb', nb_model), ('lr', lr_model)],
        final_estimator=meta_model,
        cv=5
    )
    stack_model.fit(X_train, Y_train)
    if not os.path.exists('model'):
        os.makedirs('model')
    joblib.dump(stack_model, 'model/heart_disease_model.pkl')
    return stack_model

model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        input_data = np.asarray(data).reshape(1, -1)
        prediction = model.predict(input_data)
        if prediction[0] == 0:
            result = "The Person does NOT have Heart Disease"
        else:
            result = "The Person HAS Heart Disease"
        return render_template('index.html', prediction_text=result)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
