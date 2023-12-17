import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_and_predict(test_data, model):
    # Create a DataFrame with the provided test data
    test_df = pd.DataFrame([test_data], columns=[
        'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
        'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
        'age', 'gender', 'ethnicity', 'jaundice', 'austim', 'contry_of_res', 'used_app_before',
        'result', 'age_desc', 'relation', 'Class/ASD'
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

# Load the model
model = pickle.load(open('savedmodel.sav', 'rb'))

# Test data
#test_values = [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 23.73476684, 'm', 'Hispanic', 'no', 'no', 'India', 'no', 6.495259913, '18 and more', 'Self', 0]
#test_values = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7.380373076, 'm','White-European', 'no', 'yes', 'United States', 'no', 14.85148447, '18 and more', 'Self', 1]
test_values = [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 7.380373076, 'm','White-European', 'no', 'yes', 'United States', 'no', 14.85148447, '18 and more', 'Self', 0]
# Make prediction
result = preprocess_and_predict(test_values, model)

# Print the result
print("Result: Yes" if result[0] == 1 else "Result: No")
