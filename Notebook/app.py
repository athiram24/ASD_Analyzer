from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

# Load model
model = pickle.load(open('savedmodel.sav', 'rb'))

def preprocess_and_predict(test_data, model):
    # Create a DataFrame with the provided test data
    test_df = pd.DataFrame([test_data], columns=[
        'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
        'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
        'age', 'gender', 'ethnicity', 'jaundice', 'austim', 'contry_of_res', 'used_app_before',
        'result', 'age_desc', 'relation'
    ])

    # Drop unwanted columns
    columns_to_drop = test_df.columns.difference([
        'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
        'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
        'age', 'gender', 'jaundice', 'austim', 'contry_of_res', 'result'
    ])
    test_df = test_df.drop(columns=columns_to_drop)

    # Label Encoding for categorical columns
    label_encode_cols = ['gender', 'jaundice', 'austim', 'contry_of_res']
    label_encoder = LabelEncoder()

    for col in label_encode_cols:
        test_df[col] = label_encoder.fit_transform(test_df[col])

    # Convert to int64
    test_df[label_encode_cols] = test_df[label_encode_cols].astype('int64')

    # Standard Scaling for numerical columns
    standard_scale_cols = ['age', 'result']
    scaler = StandardScaler()
    test_df[standard_scale_cols] = scaler.fit_transform(test_df[standard_scale_cols])

    # Check if the model is fitted, and if not, fit it
    if not hasattr(model, 'classes_'):
        # Replace the following line with your actual training data and labels
        # model.fit(X_train, y_train)
        pass

    # Make predictions using the provided model
    predictions = model.predict(test_df)

    return predictions

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Accept all the values from the form
        attributes = [
            'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
            'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
            'age', 'gender', 'ethnicity', 'jaundice', 'austim', 'contry_of_res',
            'used_app_before', 'result', 'age_desc', 'relation'
        ]

        values = [request.form[attr] for attr in attributes]

        # Make predictions using the model
        result = preprocess_and_predict(values, model)
        result_label = 'Yes' if result[0] == 1 else 'No'

        return render_template('index.html', result=result_label)

if __name__ == '__main__':
    app.run(debug=True)
