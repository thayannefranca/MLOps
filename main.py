from flask import Flask, request, jsonify
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import pickle

modelo = pickle.load(open('modelo.sav', 'rb'))
colunas = ['tamanho', 'ano', 'garagem']

app = Flask(__name__)

@app.route('/')
def home():
    return "MInha primeira API"

@app.route('/sentimento/<frase>')

def sentimento(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(from_lang='pt_br', to='en')
    polaridade = tb_en.sentiment.polarity

    return "polaridade: {}".format(polaridade)

@app.route('/cotacao/', methods=['POST'])
def cotacao():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = modelo.predict([dados_input])
    return jsonify(preco=preco[0])

app.run(debug=True)