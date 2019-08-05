# encoding: utf-8
"""
@author: andy
@contact: andy_viky@163.com
@github: https://github.com/AndyandViky
@csdn: https://blog.csdn.net/AndyViky
@file: definitions.py
@time: 2019/6/20 下午4:01
@desc: global var
"""

import os

# Local directory of CypherCat API
DEC_DIR = os.path.dirname(os.path.abspath(__file__))

# Local directory containing entire repo
REPO_DIR = os.path.split(DEC_DIR)[0]

# Local directory for datasets
DATASETS_DIR = os.path.join(REPO_DIR, 'datasets')

# Local directory for runs
RUNS_DIR = os.path.join(REPO_DIR, 'runs')
