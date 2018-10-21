"""
this module contains general utility codes that are used in other modules
"""

import os
import glob
import subprocess
import shlex
import time
import datetime
import json
import csv
import numpy as np
import pickle
# import pandas as pd
from functools import wraps
# from https://github.com/jfrelinger/cython-munkres-wrapper
from munkres import munkres
import constants as cs


# base classes
class MyNumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


# functions
def join_path(*args):
    return os.path.join(*args)


def give_cwd():
    return os.getcwd()


def change_cwd(path=cs.BASE_PATH):
    return os.chdir(path)


def list_all(path=cs.BASE_PATH):
    return glob.glob(os.path.join(path, '*'))


def list_files(path=cs.BASE_PATH):
    files = glob.glob(os.path.join(path, '*'))
    files = [x for x in files if os.path.isfile(x)]
    return files


def list_dirs(path=cs.BASE_PATH):
    files = glob.glob(os.path.join(path, '*'))
    files = [x for x in files if os.path.isdir(x)]
    return files


def file_exists(file_name, path_name=cs.BASE_PATH):
    return os.path.isfile(os.path.join(path_name, file_name))


def files_exist(file_names, path_name=cs.BASE_PATH):
    return all([file_exists(x, path_name) for x in file_names])


def write_bytes(byte_obj, file_path):
    with open(file_path, 'wb') as outfile:
        outfile.write(byte_obj)


def write_csv(obj, file_path):
    with open(file_path, 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(obj)


def write_json(json_obj, file_path):
    with open(file_path, 'w') as outfile:
        json.dump(json_obj, outfile, sort_keys=True, indent=4,
                  cls=MyNumpyEncoder)


def load_json(file_path):
    with open(file_path, 'r') as infile:
        return json.load(infile)


def write_np(np_obj, file_path):
    with open(file_path, 'wb') as outfile:
        np.save(outfile, np_obj)


def load_np(file_path):
    with open(file_path, 'rb') as infile:
        return np.load(infile)


def normalize(arr):
    return_val = (arr / sum(arr))
    return_val[np.isnan(return_val)] = 0
    return return_val


def time_str(mode='abs', base=None):
    if mode == 'rel':
        return str(datetime.timedelta(seconds=(time.time() - base)))
    if mode == 'raw':
        return time.time()
    if mode == 'abs':
        return time.asctime(time.localtime(time.time()))


def print_log(message, mode='info'):
    if mode == 'info':
        print ('{}: [INFO] {}'.format(time_str(), message))
    if mode == 'err':
        print ('{}: [ERROR] {}'.format(time_str(), message))
        quit()
    if mode == 'progress':
        print (' ' * 79, end="\r")
        print ('{}: [PROGRESS] {}'.format(time_str(), message), end="\r")
    if mode == 'end_progress':
        print (' ' * 79, end="\r")
        print ('{}: [PROGRESS RESULT] {}'.format(time_str(), message))


# timer wrapper
def time_it(func):
    @wraps(func)
    def timed(*args, **kw):
        start_time = time_str('raw')
        return_val = func(*args, **kw)
        end_time = time_str('raw')

        duration = str(datetime.timedelta(seconds=(end_time - start_time)))

        message = 'Function <{}({},{})> executed in {}'.format(func.__name__,
                                                               args,
                                                               kw,
                                                               duration[:-4])
        print_log(message)

        return return_val
    return timed


@time_it
def run_cmd(cmd, input=None, cwd=give_cwd()):
    process = subprocess.Popen(shlex.split(cmd),
                               shell=False,
                               stdout=subprocess.PIPE,
                               cwd=cwd)
    return process.communicate(input)


@time_it
def rename_file(file_name, new_name, path=cs.OBJ_PATH):
    run_cmd('mv {} {}'.format(
        join_path(path, file_name),
        join_path(path, new_name)))


@time_it
def save_object(obj, file_name, rewrite=False, path=cs.OBJ_PATH):
    if rewrite or not file_exists(file_name, path):
        file_path = join_path(path, file_name)
        with open(file_path, 'wb') as output:
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


@time_it
def load_object(file_name, path=cs.OBJ_PATH):
    file_path = join_path(path, file_name)
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj


# @time_it
def my_munkres(scores):
    # add small random noise!
    random_noise = cs.MUNKRES_RANDOM_NOISE * np.random.rand(*np.shape(scores))
    return munkres(scores + random_noise)


# @time_it
def linear_sum_assignment(scores, check=True, file_name=None, path_name=cs.JSON_PATH):
    if (check and (file_name is not None) and
            (file_exists(file_name, path_name))):
        return load_json(join_path(path_name, file_name))

    pair_matrix = my_munkres(scores)
    p1, p2 = [], []

    for i1, x in enumerate(pair_matrix):
        try:
            i2 = list(x).index(True)
            p1.append(i1)
            p2.append(i2)
        except ValueError:
            continue
    if file_name is not None:
        write_json((p1, p2), join_path(path_name, file_name))
    return p1, p2


# @time_it
def greedy_assignment(scores):
    slist = []
    l1, l2 = np.shape(scores)
    for i1 in range(l1):
        for i2 in range(l2):
            slist.append([i1, i2, scores[i1, i2]])
    slist.sort(key=lambda x: x[2])
    p1 = []
    p2 = []
    nodes1 = set()
    nodes2 = set()
    for score in slist:
        n1, n2, s = score
        if (n1 not in nodes1) and (n2 not in nodes2):
            nodes1.add(n1)
            nodes2.add(n2)
            # pairs.append(score)
            p1.append(n1)
            p2.append(n2)
        if ((len(nodes1) == l1) or (len(nodes2) == l2)):
            break
    return p1, p2
