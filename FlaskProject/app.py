from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        feature = [float(i) for i in request.form.values()]
        feature = [np.array(feature)]
        y_pred = model.predict(feature)

    return render_template('index.html', predict_price= "House price is {}".format(y_pred))


if __name__ == '__main__':
    app.run(debug=True)