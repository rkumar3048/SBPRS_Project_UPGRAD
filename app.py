from flask import Flask, jsonify,  request, render_template
import pickle
from project import model_recommendation
import numpy as np


app = Flask(__name__)

model_reco = pickle.load(open('./models/reco_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if (request.method == 'POST'):
        feature = request.form.values()
        output = model_reco.predict(feature)
        return render_template('index.html', prediction_text='Top 5 Recommendations for the User \n {}'.format(output))
    else :
        return render_template('index.html')

@app.route("/predict_api", methods=['POST', 'GET'])
def predict_api():
    print(" request.method :",request.method)
    if (request.method == 'POST'):
        data = request.get_json()
        return jsonify(model_load.predict([np.array(list(data.values()))]).tolist())
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)