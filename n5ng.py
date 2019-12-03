import argparse
import gzip
import io
import zarr
import numpy as np
from flask import Flask, jsonify, Response, request, render_template
from flask_cors import CORS
from n5ContainerSpec import parse
from glob import glob
from neuroglancer.viewer_state import ViewerState
from neuroglancer import url_state

app = Flask(__name__)
CORS(app)

def get_scale_for_dataset(dataset, scale, base_res, encoding='raw'):
        if 'resolution' in dataset.attrs:
            resolution = dataset.attrs['resolution']
        elif 'downsamplingFactors' in dataset.attrs:
            # The FAFB n5 stack reports downsampling, not absolute resolution
            resolution = (base_res * np.asarray(dataset.attrs['downsamplingFactors'])).tolist()
        else:
            resolution = (base_res*2**scale).tolist()
        return {
                    'chunk_sizes': [list(reversed(dataset.chunks))],
                    'resolution': resolution,
                    'size': list(reversed(dataset.shape)),
                    'key': str(scale),
                    'encoding': encoding,
                    'voxel_offset': dataset.attrs.get('offset', [0,0,0]),
                }

def get_scales(dataset_name, scales, encoding='raw', base_res=np.array([1.0,1.0,1.0])):
    if  scales:
        # Assumed scale pyramid with the convention dataset/sN, where N is the scale level
        scale_info = []
        for scale in scales:
            try:
                dataset_name_with_scale = "%s/s%d" % (dataset_name, scale)
                dataset = app.config['n5data'][dataset_name][dataset_name_with_scale];
                scale_info.append(get_scale_for_dataset(dataset, scale, base_res, encoding))
            except Exception as exc:
                print(exc)
    else:
        print(f'No scales found for {dataset_name}.')
        dataset = app.config['n5data'][dataset_name]
        # No scale pyramid for this dataset
        scale_info = [ get_scale_for_dataset(dataset, 1.0, base_res, encoding) ]       
    print(scale_info)
    return scale_info

@app.route('/<path:dataset_name>/info')
def dataset_info(dataset_name):
    info = {
        'data_type' : 'uint8',
        'type': 'image',
        'num_channels' : 1,
        'scales' : get_scales(dataset_name, scales=list(range(0,8)), base_res=np.array([1.0, 1.0, 1.0]))
    }
    print(info)
    return jsonify(info)

# Implement the neuroglancer precomputed filename structure for URL requests
@app.route('/<path:dataset_name>/<int:scale>/<int:x1>-<int:x2>_<int:y1>-<int:y2>_<int:z1>-<int:z2>')
def get_data(dataset_name, scale, x1, x2, y1, y2, z1, z2):
    # TODO: Enforce a data size limit
    dataset_name_with_scale = f'{dataset_name}/s{scale}'
    dataset = app.config['n5files'][dataset_name][dataset_name_with_scale]
    print('Loading data from disk')
    data = dataset[z1:z2,y1:y2,x1:x2]
    # Neuroglancer expects an x,y,z array in Fortran order (e.g., z,y,x in C =)
    response = Response(data.tobytes(order='C'), mimetype='application/octet-stream')

    accept_encoding = request.headers.get('Accept-Encoding', '')
    if 'gzip' not in accept_encoding.lower() or \
           'Content-Encoding' in response.headers:
            return response

    gzip_buffer = io.BytesIO()
    gzip_file = gzip.GzipFile(mode='wb', compresslevel=5, fileobj=gzip_buffer)
    gzip_file.write(response.data)
    gzip_file.close()
    response.data = gzip_buffer.getvalue()
    response.headers['Content-Encoding'] = 'gzip'
    response.headers['Content-Length'] = len(response.data)

    return response

    
def get_datasets(directory):
    n5s = glob(directory + '*/*/*.n5')
    result = {}
    [result.update(parse(d)) for d in n5s]
    return  result

def make_neuroglancer_link(ng_server_address, dataset_name, voxel_size=[1.0, 1.0, 1.0], coords=[0.0, 0.0, 0,0], zoom=1.0):
    from collections import OrderedDict
    # need to know resolution, size of the data, name of the data
    zoom = 1.0
    flask_address = 'http://localhost:5000'
    statedict = OrderedDict([('layers',
            [OrderedDict([('source',
                            f'precomputed://{flask_address}/{dataset_name}'),
                        ('type', 'image'),
                        ('blend', 'default'),
                        ('name', dataset_name)])]),
            ('navigation',
            OrderedDict([('pose',
                        OrderedDict([('position',
                                        OrderedDict([('voxelSize',
                                                    voxel_size),
                                                    ('voxelCoordinates',
                                                    coords)])),
                                        ])),
                        ('zoomFactor', zoom)])),
            ('layout', '4panel')])

    link = url_state.to_url(ViewerState(statedict), prefix=ng_server_address)
    return link

@app.route('/')
def get_home():
    paramdict = {k: v for k, v in app.config['datasets'].items()}
    for k in app.config['neuroglancer_links'].keys():
        paramdict[k]['neuroglancer_link'] = app.config['neuroglancer_links'][k]
    return render_template('index.html', datasets=paramdict.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', help='directory to search', default=None)
    args = parser.parse_args()
    datasets = None
    if args.dir:
        datasets = get_datasets(args.dir)
    else:
        print('Nothing to do without a -d argument.')
        return None
    neuroglancer_address = 'http://localhost:8080'
    # Start flask
    app.debug = True
    app.config['datasets'] = datasets
    app.config['n5data'] = {k: zarr.open(v['container_path'], mode='r') for k, v in datasets.items()}
    app.config['neuroglancer_links'] = {k: make_neuroglancer_link(neuroglancer_address, k) for k,v in datasets.items()}
    app.config['neuroglancer_address'] = neuroglancer_address
    app.run(host='0.0.0.0')

if __name__ == '__main__':
    main()

