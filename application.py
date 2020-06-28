from flask import Flask, jsonify, request
import joblib
import pickle
import torch

app = Flask(__name__)

class LSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.fc1 = nn.Linear(hidden_dim, 100)
        self.fc2 = nn.Linear(100, output_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        _, (hidden_text, _) = self.lstm(embeds.view(len(sentence), 1, -1))
        hidden_text = nn.ReLU()(hidden_text)
        prediction = self.fc1(hidden_text.squeeze(0))
        prediction = nn.ReLU()(prediction)
        prediction = self.fc2(prediction)
        return prediction

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/score", methods=['POST'])
def score():
    data = request.json
    print(data)
    print(data['text'])
    model_path = './FINAL_LSTM.txt'
    model = LSTM(16, 16, 339, 5)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return jsonify({"score": 1})