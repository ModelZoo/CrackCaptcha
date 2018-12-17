import json
from flask import Flask, render_template, request
import logging
from flask_cors import CORS as cors

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
app = Flask(__name__)

from os.path import join, exists, isdir
from os import listdir, makedirs
import re
from labeler.config import DATA_MAP as data_map, DEFAULT_DATASET as default_dataset, LIMIT_PER_PAGE as limit

cors(app)


@app.route('/')
def index():
    dataset = request.args.get('dataset', default_dataset)
    return render_template(f'{dataset}.html')


@app.route('/data', methods=['GET', 'POST'])
def data():
    result = []
    dataset = request.args.get('dataset', default_dataset)
    page = int(request.args.get('page', 1))
    
    for file in listdir(join('data', dataset)):
        if re.match(data_map[type]['image'], file):
            result.append(file)
    # sort result
    result = sorted(result)
    # slice result
    result = result[(page - 1) * limit:page * limit]
    return json.dumps(result)


@app.route('/update', methods=['GET', 'POST'])
def update():
    label = request.form['label']
    label_file = request.form['label_file']
    with open(label_file, 'w', encoding='utf-8') as f:
        f.write(label)
    checked_file = label_file.replace('.label', '.checked.filter')
    with open(checked_file, 'w', encoding='utf-8') as f:
        f.write('1')
    return json.dumps({
        'result': 1
    })


# if __name__ == '__main__':
def run():
    app.run(debug=True,
            host='0.0.0.0',
            # ssl_context='adhoc',
            port=5000)
