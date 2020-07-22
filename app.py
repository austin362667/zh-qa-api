#coding:utf-8
from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import calendar
import time

app = Flask(__name__)
model_path = './starkCache.pkl'
device = torch.device("cpu")



@app.route('/')
def hello_world():
  return 'Hello, World!'

@app.route('/healthcheck', methods=['POST'])
def healthcheck():
    req_json = request.get_json()
    print(req_json)

    esun_uuid = req_json['esun_uuid']
    server_uuid = '0000000001'
    captain_email  = 'austin362667@gmail.com'
    server_timestamp = calendar.timegm(time.gmtime())

    res_json = { "esun_uuid": esun_uuid,"server_uuid": server_uuid,"captain_email": captain_email,"server_timestamp": server_timestamp,}
    print(res_json) 
    return res_json


@app.route('/inference', methods=['POST'])
def inference():
    req_json = request.get_json()
    print(req_json)

    esun_uuid = req_json['esun_uuid']
    server_uuid = '0000000001'
    news  = req_json['news']
    print(news)
    server_timestamp = calendar.timegm(time.gmtime())
    answer = []
    answer.append(inference_engine(news))

    res_json = { "esun_uuid": esun_uuid,"server_uuid": server_uuid,"server_timestamp": server_timestamp,"answer": answer,}
    print(res_json) 
    return res_json



def inference_engine(content):
    
    model.eval()
    ans_lst = []
    
    text = content
    if len(text) > 400:
        text = text[:400]
    question = u"""
                洗錢是指將犯罪不法所得，掩飾、隱匿非法金融行為。
                國內吸金、電信詐騙案件也層出不窮，請問哪些人有可能在洗錢?
            """
    encoding = tokenizer.encode_plus(question, text)
    input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]
    start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer = ''.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
    
    if len(answer) > 15 or len(answer) == 1 or answer == '洗錢':
        answer = ''

    return answer

import os
if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    model = BertForQuestionAnswering.from_pretrained('hfl/chinese-roberta-wwm-ext')
    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    app.debug = True
    app.run(host='0.0.0.0', port=443)
