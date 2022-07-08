import sys
import os
from os import path as osp


this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, '..', 'lib')

if lib_path not in sys.path:
    sys.path.insert(0, lib_path)
    try:
        os.environ["PYTHONPATH"] = lib_path + ":" + os.environ["PYTHONPATH"]
    except KeyError:
        os.environ["PYTHONPATH"] = lib_path
