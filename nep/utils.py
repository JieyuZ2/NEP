import collections
import random
import time
import csv
import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
from decimal import *
from collections import defaultdict
from scipy.sparse import csr_matrix
from tqdm import tqdm, trange
from random import shuffle
import torch
# matplotlib.use('Agg')


def print_config(config, logger=None):
    config = vars(config)
    info = "Running with the following configs:\n"
    for k, v in config.items():
        to_add = "\t{} : {}\n".format(k, str(v))
        if len(to_add) < 1000:
            info += to_add
    info.rstrip()
    if not logger:
        print("\n" + info)
    else:
        logger.info("\n" + info)


def exec_time(func):
    def new_func(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("Cost {} seconds.".format(end - start))
        return result

    return new_func


def save_checkpoint(state, modelpath, modelname, logger=None, del_others=True):
    if del_others:
        for dirpath, dirnames, filenames in os.walk(modelpath):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                if path.endswith('pth.tar'):
                    if logger is None:
                        print(f'rm {path}')
                    else:
                        logger.warning(f'rm {path}')
                    os.system("rm -rf '{}'".format(path))
            break
    path = os.path.join(modelpath, modelname)
    if logger is None:
        print('saving model to {}...'.format(path))
    else:
        logger.warning('saving model to {}...'.format(path))
    torch.save(state, path)


def load_node(path):
    """
    :param path: node.dat
    :return:
    """
    name_to_id, id_to_name = {}, []
    type_to_node, node_to_type = defaultdict(list), []
    node_id, type_id = 0, 0
    with open(path) as f:
        for raw_line in f:
            line = raw_line.rstrip().split('\t')
            name_to_id[line[0]] = node_id
            id_to_name.append(line[0])
            type_to_node[line[1]].append(node_id)
            node_to_type.append(line[1])
            node_id += 1

    type_to_node_copy = {}
    for type, ids in type_to_node.items():
        type_to_node_copy[type] = np.array(ids)

    type_id = 0
    type_to_id, id_to_type = {}, []
    for id, t in enumerate(type_to_node.keys()):
        type_to_id[t] = id
        id_to_type.append(t)
        type_id += 1
    return id_to_name, name_to_id, node_to_type, type_to_node_copy, id_to_type, type_to_id


def load_label(path):
    with open(path, 'r') as inf:
        labels = set()
        name_to_label = {}
        for l in inf:
            data, label = l.rstrip().split('\t')
            labels.add(int(label))
            name_to_label[data] = int(label)
    return name_to_label, len(labels)


def load_graph(name_to_id, num_node, path):
    row_col_to_links = defaultdict(list)
    link_type = set()
    neighbor_by_link_dict = defaultdict(dict)
    with open(path) as f:
        num_link = 0
        for line in f:
            line = line.rstrip().split('\t')
            if len(line) == 3:
                num_link += 1
                s, e = name_to_id[line[0]], name_to_id[line[1]]
                row_col_pair = (s, e)
                d = int(line[2])
                link_type.add(d)
                row_col_to_links[row_col_pair].append(d)
                if d in neighbor_by_link_dict[s]:
                    neighbor_by_link_dict[s][d].append(e)
                else:
                    neighbor_by_link_dict[s][d] = [e]
    num_link_type = len(link_type)

    for node, links in neighbor_by_link_dict.items():
        for link, neighbors in links.items():
            shuffle(neighbor_by_link_dict[node][link])

    id_to_links = [None]  # because link id starts from 1
    link_to_ids = defaultdict(list)
    rows, cols, data = [], [], []
    for (row, col), links in row_col_to_links.items():
        sorted_links = sorted(links)
        rows.append(row)
        cols.append(col)
        if sorted_links not in id_to_links:
            id_to_links.append(sorted_links)
            links_id = len(id_to_links)-1
            for link in links:
                link_to_ids[link].append(links_id)
        else:
            links_id = id_to_links.index(sorted_links)
        data.append(links_id)

    return csr_matrix((data, (rows, cols)), shape=(num_node, num_node)), link_type, num_link_type, num_link, id_to_links, link_to_ids, neighbor_by_link_dict


def load_train_test_node(dir):
    test_nodes, train_nodes = [],[]
    node_to_label = defaultdict(int)

    with open(osp.join(dir, 'train_nodes.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for line in csv_reader:
            train_nodes.append(line[0])
            node_to_label[line[0]] = line[1]
    with open(osp.join(dir, 'test_nodes.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for line in csv_reader:
            test_nodes.append(line[0])
            node_to_label[line[0]] = line[1]
    # with open(osp.join(dir, 'all_nodes.csv')) as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=',')
    #     for line in csv_reader:
    #         node_to_label[line[0]] = line[1]

    return train_nodes, test_nodes, node_to_label


def load_feature(path):
    feature = {}
    dim = -1
    with open(path) as f:
        for line in f:
            line = line.rstrip().split(',', 1)
            vector = list(map(float, line[1].split(',')))
            dim = len(vector)
            feature[line[0]] = np.array(vector)
    return feature, dim


def load_groups(paths):
    groups = []
    for path in paths:
        group = set()
        with open(path) as f:
            for line in f:
                group.add(line.rstrip())
        groups.append(group)
    return groups


def next_batch(X, y, batch_size, shuffle=True, choice=None):
    if choice is not None:
        X = X[choice]
        y = y[choice]
    num = len(X)
    if y is None:
        if shuffle:
            permutation = np.random.permutation(num)
            X = X[permutation]
        for i in np.arange(0, num, batch_size):
            yield X[i:i + batch_size]
    else:
        if shuffle:
            permutation = np.random.permutation(num)
            X = X[permutation]
            y = y[permutation]
        for i in np.arange(0, num, batch_size):
            yield X[i:i + batch_size], y[i:i + batch_size]


def join_int(l):
    return ','.join([str(i) for i in l])


def write_train_data(file_name, data):
    with open(file_name, 'w') as f:
        paths, labels = data
        for i,path in enumerate(paths):
            f.write(join_int(path)+'\t'+join_int(labels[i])+'\n')


def load_train_data(file_path):
    paths, labels = [],[]
    if file_path[-3:] == 'txt':
        with open(file_path, 'r') as f:
            for line in tqdm(f, desc='reading training data from file:'+file_path):
                path, label = line.strip().split('\t')
                paths.append([int(i) for i in path.split(',')])
                labels.append([int(i) for i in label.split(',')])
        labels = np.array(labels)
        paths = np.array(paths)
    elif file_path[-3:] == 'npz':
        print('load training data from file:'+file_path)
        data = np.load(file_path)
        paths = data['paths']
        labels = data['labels']
    else:
        print('ERROR: unsupport data file!')
        exit(1)
    return paths, labels


def plot(data, plot_file):
    plt.figure()
    plt.plot(range(len(data)), data)
    plt.savefig(plot_file)
    plt.close()


def load_graph_new(name_to_id, path):
    link_type = set()
    link_dict = defaultdict(list)
    with open(path) as f:
        for line in tqdm(f, desc='loading link data'):
            line = line.rstrip().split('\t')
            if len(line) == 3:
                s, e = name_to_id[line[0]], name_to_id[line[1]]
                row_col_pair = (s, e)
                d = int(line[2])
                link_type.add(d)
                link_dict[d].append(row_col_pair)
    return link_type, link_dict


class VoseAlias(object):
    """
    Adding a few modifs to https://github.com/asmith26/Vose-Alias-Method
    """

    def __init__(self, dist):
        """
        (VoseAlias, dict) -> NoneType
        """
        self.dist = dist
        self.alias_initialisation()

    def alias_initialisation(self):
        """
        Construct probability and alias tables for the distribution.
        """
        # Initialise variables
        n = len(self.dist)
        self.table_prob = {}   # probability table
        self.table_alias = {}  # alias table
        scaled_prob = {}       # scaled probabilities
        small = []             # stack for probabilities smaller that 1
        large = []             # stack for probabilities greater than or equal to 1

        # Construct and sort the scaled probabilities into their appropriate stacks
        # print("1/2. Building and sorting scaled probabilities for alias table...")
        for o, p in self.dist.items():
            scaled_prob[o] = Decimal(p) * n

            if scaled_prob[o] < 1:
                small.append(o)
            else:
                large.append(o)

        # print("2/2. Building alias table...")
        # Construct the probability and alias tables
        while small and large:
            s = small.pop()
            l = large.pop()

            self.table_prob[s] = scaled_prob[s]
            self.table_alias[s] = l

            scaled_prob[l] = (scaled_prob[l] + scaled_prob[s]) - Decimal(1)

            if scaled_prob[l] < 1:
                small.append(l)
            else:
                large.append(l)

        # The remaining outcomes (of one stack) must have probability 1
        while large:
            self.table_prob[large.pop()] = Decimal(1)

        while small:
            self.table_prob[small.pop()] = Decimal(1)
        self.listprobs = list(self.table_prob)

    def alias_generation(self):
        """
        Yields a random outcome from the distribution.
        """
        # Determine which column of table_prob to inspect
        col = random.choice(self.listprobs)
        # Determine which outcome to pick in that column
        if self.table_prob[col] >= random.uniform(0, 1):
            return col
        else:
            return self.table_alias[col]

    def sample_n(self, size):
        """
        Yields a sample of size n from the distribution, and print the results to stdout.
        """
        for i in range(size):
            yield self.alias_generation()


def makeDist(graphpath, node2idx, power=0.75):
    nodedistdict = collections.defaultdict(int)

    weightsdict = collections.defaultdict(int)
    nodedegrees = collections.defaultdict(int)

    negprobsum = 0
    nlines = 0

    with open(graphpath, "r") as graphfile:
        for l in graphfile:
            nlines += 1

    maxindex = 0
    with open(graphpath, "r") as graphfile:
        for l in graphfile:

            line = l.rstrip().split('\t')
            node1, node2 = node2idx[line[0]], node2idx[line[1]]

            nodedistdict[node1] += 1
            nodedegrees[node1] += 1
            negprobsum += np.power(1, power)

            if node1 > maxindex:
                maxindex = node1
            elif node2 > maxindex:
                maxindex = node2

    for node, outdegree in nodedistdict.items():
        nodedistdict[node] = np.power(outdegree, power) / negprobsum

    return nodedistdict


def negSampleBatch(sourcenode, targetnode, negsamplesize, nodesaliassampler, t=10e-3):
    """
    For generating negative samples.
    """
    negsamples = 0
    while negsamples < negsamplesize:
        samplednode = nodesaliassampler.sample_n(1)
        if (samplednode == sourcenode) or (samplednode == targetnode):
            continue
        else:
            negsamples += 1
            yield samplednode


if __name__ == '__main__':
    ratio = 0.8

    # ds = data_loader(data_dir='data/dblp_data/full_graph', train_test_dir='data/dblp_data/train_test_node_ratio={}'.format(ratio))

    # ds = data_loader(data_dir='data/yago/graph_wasBornIn', train_test_dir='data/yago/graph_wasBornIn/train_test_node_ratio={}'.format(ratio))
