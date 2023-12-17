
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open('savedmodel.sav', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/Check', methods=['POST'])
def Check():
    
    if request.method == 'POST':
        # Accept all the values from the form
        attributes = [
            'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
            'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
            'age', 'gender', 'jaundice', 'austim', 'contry_of_res',
            'result'
        ]

        values = []
        for attribute in attributes:
            if attribute in ['gender', 'jaundice', 'austim']:
                # Assuming binary values where 'yes' is encoded as 1 and 'no' as 0
                values.append(1 if request.form[attribute] == 'yes' else 0)
            else:
                # For other attributes, assume they are numeric
                values.append(float(request.form[attribute]))

        # Make predictions using the model
        result = model.predict([values])[0]
       
        
        print(result)
      
        return result

if __name__ == '__main__':
    app.run(debug=True)

