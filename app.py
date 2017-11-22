from flask import Flask, jsonify, request
import sys
from sklearn import datasets, svm
import numpy as np
import lasagne
from keras.models import load_model
from flask_compress import Compress
from rnn_model.neural_networks.rnn_one_hot import RNNOneHot
from rnn_model.neural_networks.helpers.dummy_data_handling import DummyDataHandler
from rnn_model.neural_networks.update_manager import Adam
from rnn_model.neural_networks.target_selection import SelectTargets
from rnn_model.neural_networks.sequence_noise import SequenceNoise
from rnn_model.neural_networks.recurrent_layers import RecurrentLayers

app = Flask(__name__)
Compress(app)
# model = load_model("./model/model.hf5", compile=False)
# updater = Adam(learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999)
# target_selection=SelectTargets(n_targets=1, shuffle=False, bias=-1.0, determinist_test=True)
# sequence_noise = SequenceNoise(dropout=0.0, swap=0.0, ratings_perturb=0.0, shuf=0.0, shuf_std=5.0)
# recurrent_layer = RecurrentLayers(layer_type='GRU', layers=[50], bidirectional=False, embedding_size=0)

# predictor = RNNOneHot(interactions_are_unique=True, max_length=30, diversity_bias=0.0, regularization=0.0, updater=updater, target_selection=target_selection, sequence_noise=sequence_noise, recurrent_layer=recurrent_layer, use_ratings_features=False, use_movies_features=False, use_users_features=False, batch_size=16)
# dummy_dataset = DummyDataHandler(n_items=10681)
# predictor.prepare_model(dummy_dataset)
# predictor.load("./rnn_model/model.999_nt1_nf")

def rnn_predict(json):
    formatted = [[item['id'], item['rating']] for item in json]
    return [x.item() for x in predictor.top_k_recommendations(formatted)]

def ml_predict(json):
    ratings = np.zeros(10681)
    json_ids = [x['id'] for x in json]
    print('json ids', file=sys.stderr)
    print(json_ids, file=sys.stderr)

    for i in json:
        ratings[i['id']] = i['rating']
    predictions = model.predict(np.expand_dims(ratings, 0))[0]
    predictions = [{'id': i, 'predicted_rating': np.float64(x)} for i,x in enumerate(predictions)]
    predictions.sort(key= lambda x: x['predicted_rating'], reverse=True)
    returned_predictions = []
    count = 0
    for p in predictions:
        returned_predictions.append(p)
        if(p['id'] not in json_ids):
           count += 1
        if(count >= 40):
            break

    return returned_predictions

@app.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    header['Access-Control-Allow-Headers'] = 'Accept,Content-Type'
    return response

@app.route('/predict', methods=['POST'])
def predict():
    json = request.json
    prediction = ml_predict(json)
    next_recs = rnn_predict(json)

    return jsonify({'predictions': prediction, 'next_recs': next_recs}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0')
