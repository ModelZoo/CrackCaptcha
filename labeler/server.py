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

cors(app)


@app.route('/')
def index():
    return render_template('check.html')


@app.route('/data', methods=['GET', 'POST'])
def data():
    # label_file =
    pass

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


if __name__ == '__main__':
    app.run(debug=True,
            host='0.0.0.0',
            # ssl_context='adhoc',
            port=5557)
