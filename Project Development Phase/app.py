from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('best_random_forest_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    num_attributes = 64
    attr_values_str = request.form['attributes']
    attr_values_list = [float(value.strip()) for value in attr_values_str.split(',')]


    # Ensure there are exactly 64 attributes
    if len(attr_values_list) != num_attributes:
        return "Please enter 64 attributes separated by commas."
    
    data = [attr_values_list]
    prediction = model.predict(data)[0]
    return render_template('predict.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)