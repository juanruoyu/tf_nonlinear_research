### train by GPU or nonGPU
If you donnot have GPU and you can choose to skip extra\_32x32.mat when trainining by set use\_extra\_data = False  in config.py.

### software before training 
please install Python 3.7.0, tensorflow.
if you are MacOS use:
`pip3 install tensorflow` 
`pip3 install libmagic`

if you are ubuntu users use:
`pip3 install python-magic tabulate --user`
`pip3 install tensorflow` 

### download dataset:
 Open http://ufldl.stanford.edu/housenumbers . Please download format2 data. (train\_32x32.mat, test\_32x32.mat, extra\_32x32.mat)

### view dataset
 use `python3 dataset.py`

### start train
 use 
