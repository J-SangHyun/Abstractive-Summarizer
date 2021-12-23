# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from flask_cors import CORS
from argparse import ArgumentParser

# Import tokenizer
from tokenizer.okt.okt import OktTokenizer
from tokenizer.sentencepiece.sentencepiece import SentencePieceTokenizer

# Import seq2seq module
from seq2seq.rnn.module import RNNModule
from seq2seq.transformer.module import TransformerModule

# Import global variables
from utils.global_variables import input_max_length, target_max_length

parser = ArgumentParser()
parser.add_argument('-t', '--tokenizer', type=str, default='sentencepiece', help='choose tokenizer')
args = parser.parse_args()

# Set tokenizer
if args.tokenizer == 'okt':
    tokenizer = OktTokenizer()
elif args.tokenizer == 'sentencepiece':
    tokenizer = SentencePieceTokenizer()
else:
    raise NameError(f'tokenizer name {args.tokenizer} is not defined.')
print(f'Tokenizer Model: {args.tokenizer}')

# Set seq2seq module
rnn_module = RNNModule(tokenizer)
transformer_module = TransformerModule(tokenizer)

app = Flask(__name__)
CORS(app)


def get_prediction(input_sentence, module):
    indexes = tokenizer.encode(input_sentence, max_length=input_max_length)
    predicted_indexes = module.predict(indexes, max_length=target_max_length)
    return tokenizer.decode(predicted_indexes)


# Test summarizer
for m in [rnn_module, transformer_module]:
    get_prediction('test', module=m)
    print(f'{m.name} module working.')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_sentence = request.form['input_sentence']
        return jsonify({'rnn': get_prediction(input_sentence, rnn_module),
                        'transformer': get_prediction(input_sentence, transformer_module)})


if __name__ == '__main__':
    app.run()
