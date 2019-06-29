# DEC-pytorch
---
Using dec(deep embedded cluster) to get 'n' cluster in Minist dataset.

Requirement
python 3.5
argparse
pickle
pytorch == 1.1.0
numpy == 1.12.1
sklearn == 0.18.1

File
datasets # container of data
dec # dec core code file
runs # the result of runing
train.py # train code

How to run code?
params:
-r Name of training run
-p Number of epochs
-b Number of batch size
-s Name if dataset default is minist

example:
python train.py -p 200000 -b 60 -s minist
