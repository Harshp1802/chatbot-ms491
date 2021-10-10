import warnings
warnings.filterwarnings('ignore')


from flask import Flask, render_template, request
from flask import jsonify
from flask_wtf.csrf import CSRFProtect

import sys
import os

from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
mname = 'facebook/blenderbot-400M-distill'
model = BlenderbotForConditionalGeneration.from_pretrained(mname)
tokenizer = BlenderbotTokenizer.from_pretrained(mname)

def inference(MSG):
	UTTERANCE = (MSG)
	print(UTTERANCE)
	inputs = tokenizer([UTTERANCE], return_tensors='pt')
	# reply_ids = model.generate(**inputs)
	# reply_ids = model.generate(**inputs, max_length=100,num_beams=3, no_repeat_ngram_size=2, early_stopping=True)
	reply_ids = model.generate(**inputs, max_length=100,top_k=50,top_p=0.95, no_repeat_ngram_size=2, early_stopping=True)
	print(tokenizer.batch_decode(reply_ids,skip_special_tokens=True))
	return (tokenizer.batch_decode(reply_ids,skip_special_tokens=True))


app = Flask(__name__,static_url_path="/static")
csrf = CSRFProtect(app)
csrf.init_app(app)

@csrf.exempt
@app.route('/test', methods=['POST'])

def reply():
	global COUNT,HISTORY
	COUNT+=1
	print(COUNT)
	if(COUNT%3==0):
		HISTORY = ''
		COUNT=0
	json1 = request.json
	text = json1['msg']
	HISTORY += text  + '    '
	output = inference(text)
	HISTORY += text  + '    '
	return jsonify(text=output)


@app.route("/")
def index():
    return render_template("index.html")

if (__name__ == "__main__"):
	global COUNT,HISTORY
	COUNT = 0
	HISTORY = ''
	app.run(port = 5000,debug=False)
