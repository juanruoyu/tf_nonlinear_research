### train by GPU or nonGPU
If you donnot have GPU and you can choose to skip extra\_32x32.mat when trainining by set use\_extra\_data = False  in common.py.

### software before training 
my os is Mac OS and install Python 3.7.0, tensorflow.
use:
`pip3 install tensorflow` 
`pip3 install libmagic`

### download dataset:
 Open http://ufldl.stanford.edu/housenumbers . Please download format2 data. (train\_32x32.mat, test\_32x32.mat, extra\_32x32.mat)

### view dataset
 use `python3 dataset.py`
