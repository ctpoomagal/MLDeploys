from flask import request, jsonify, Flask
import pandas as pd # Import pandas to convert input to DataFrame
import joblib
app = Flask(__name__)


# Define file paths for loading the model and scaler
model_filename = 'logistic_regression_model.pkl'
scaler_filename = 'standard_scaler.pkl'

# Load the trained Logistic Regression model
log_reg_model = joblib.load(model_filename)
print(f"Logistic Regression model loaded from {model_filename}")

# Load the fitted StandardScaler
scaler = joblib.load(scaler_filename)
print(f"StandardScaler loaded from {scaler_filename}")
# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if not request.json:
        return jsonify({'error': 'Invalid input, request must be JSON'}), 400

    data = request.json

    # The input data should be a dictionary matching the feature names
    # For this model, the features are: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
    try:
        # Convert input data to a DataFrame, ensuring the order of columns matches training data
        # This assumes the input JSON will have keys matching feature names
        input_df = pd.DataFrame([data])

        # Ensure columns are in the correct order as expected by the scaler and model
        # The order of columns from X (features) was:
        # ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        # Make sure input_df has these columns in this order
        feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        input_df = input_df[feature_columns]

        # Scale the input data
        scaled_input = scaler.transform(input_df)

        # Make a prediction
        prediction = log_reg_model.predict(scaled_input)[0]
        prediction_proba = log_reg_model.predict_proba(scaled_input)[0][1] # Probability of class 1 (diabetes)

        # Map prediction to a readable format
        result = 'Diabetic' if prediction == 1 else 'Non-Diabetic'

        return jsonify({
            'prediction': result,
            'probability': float(prediction_proba)
        })
    except KeyError as e:
        return jsonify({'error': f'Missing input feature: {e}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid input value: {e}'}), 400
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {e}'}), 500

print("Prediction endpoint '/predict' defined.")

if __name__ == "__main__":
  Print("Starting prediction API with Preprocessing and model Inference......")
  app.run(debug=True)
