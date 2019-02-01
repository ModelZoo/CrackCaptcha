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
    dataset = request.args.get('dataset', default_dataset)
    page = int(request.args.get('page', 1))
    
    # find, filter, sort, slice
    files = list(listdir(join('datasets', dataset)))
    files = list(filter(lambda file: re.match(data_map[dataset]['image'], file), files))
    files = sorted(files)
    files = files[(page - 1) * limit:page * limit]
    
    # transfer to result
    result = []
    for index, file in enumerate(files):
        image_path = join('datasets', dataset, file)
        label_path = re.sub(data_map[dataset]['image'], data_map[dataset]['label'], image_path)
        print(label_path)
        label = open(label_path).read().strip() if exists(label_path) else -1
        result.append({
            'image': join('static', image_path),
            'label': float(label),
            'name': file,
            'page': page,
            'offset': index
        })
    return json.dumps(result)


@app.route('/label', methods=['GET', 'POST'])
def update():
    name = request.form['name']
    dataset = request.form['dataset']
    ratio = request.form['ratio']
    image_path = join('datasets', dataset, name)
    label_path = re.sub(data_map[dataset]['image'], data_map[dataset]['label'], image_path)
    with open(label_path, 'w', encoding='utf-8') as f:
        f.write(ratio)
    return json.dumps({
        'label': ratio,
        'success': 1
    })


# if __name__ == '__main__':
def run():
    app.run(debug=True,
            host='0.0.0.0',
            threaded=True,
            # ssl_context='adhoc',
            port=5000)
