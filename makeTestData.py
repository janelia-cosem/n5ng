import numpy as np
import zarr
import argparse
from skimage.transform import pyramid_gaussian
from numcodecs import GZip

def make_pyramids(size):
    data = np.random.randint(0, 255, size=(size,size,size)).astype('float32')
    pyr = [p.astype('uint8') for p in pyramid_gaussian(data)]
    return pyr

def save_pyramid(path, pyramid):
    compressor = None
    store = zarr.N5Store(path)
    group = zarr.open(store=store,path='data', mode='w')
    group.attrs.update({'downsamplingFactors':[2,2,2], 'resolution' : [1,1,1]})
    for ind, p in enumerate(pyramid):
        arr = group.array(name=f's{ind}', data=p, compressor=compressor,chunks=(100,100,100))
        arr.attrs.update()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', help='Name of the n5 file', default='sample.n5')
    parser.add_argument('-s', help='length of each axis of the 3D data', default=128, type=int)
    args = parser.parse_args()

    pyramid = make_pyramids(args.s)
    save_pyramid(args.f, pyramid)
    
if __name__ == '__main__':
    main()