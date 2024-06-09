import numpy as np
import struct
# 用正确的路径替换 'path/to/your/file.dat'
filename = '../experiment_data/NYX.f32'
data = np.fromfile(filename, dtype='<f4')
data.tofile("NYX.bin")