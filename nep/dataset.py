import json
from collections import Counter
from torch.utils.data import Dataset
from nep.utils import *


class EvaDataset(Dataset):
    def __init__(self, X, y):
        self.len = len(X)
        self.data = [(X[i], y[i]) for i in range(self.len)]

    def __getitem__(self, index):
        batch, label = self.data[index]
        return batch, label

    def __len__(self):
        return self.len


class Dataset(object):
    def __init__(self, data_dir, num_data_per_epoch, threshold, train_ratio=0.8, superv_ratio=1.0):
        self.data_dir = data_dir
        train_test_dir = osp.join(data_dir, 'train_test_node_ratio=0.8')
        self.node_file = osp.join(data_dir, 'node.dat')
        self.link_file = osp.join(data_dir, 'link.dat')
        self.label_file = osp.join(self.data_dir, 'label.dat')

        #load nodes
        id_to_name, self.name_to_id, node_to_type, self.type_to_node, id_to_type, self.type_to_id = \
            load_node(self.node_file)
        self.id_to_name = np.array(id_to_name)
        self.node_to_type = np.array(node_to_type)
        self.id_to_type = np.array(id_to_type)
        self.num_node = len(self.id_to_name)
        self.num_type = len(self.id_to_type)

        #build graph
        self.graph, self.link_types, self.num_link_type, self.num_link, self.id_to_links, self.link_to_ids, self.neighbor_by_link_dict \
            = load_graph(self.name_to_id, self.num_node, self.link_file)

        # load splited labeled data
        train_nodes, test_nodes, node_to_label = load_train_test_node(train_test_dir)
        self.train_nodes = np.array([self.name_to_id[node] for node in train_nodes])
        self.test_nodes= np.array([self.name_to_id[node] for node in test_nodes])
        self.node_to_label = {self.name_to_id[node]:int(label)-1 for node, label in node_to_label.items()}
        self.num_class = len(set(list(node_to_label.values())))

        # load labeled data
        # name_to_label, self.num_class = load_label(self.label_file)
        # self.node_to_label = {self.name_to_id[name]:label for name, label in name_to_label.items()}
        # labeled_data = list(self.node_to_label.keys())
        # shuffle(labeled_data)
        # split = int(len(name_to_label)*float(train_ratio))
        # self.train_nodes = np.array(labeled_data[:int(split*float(superv_ratio))])
        # self.test_nodes= np.array(labeled_data[split:])

        self.data_threshold = max(int((num_data_per_epoch * threshold) // len(self.train_nodes)), 1)

    def free_memory(self):
        del self.graph
        del self.id_to_links
        del self.link_to_ids
        del self.name_to_id
        del self.node_to_type
        del self.type_to_id
        del self.id_to_type

    def get_neighbors(self, node, num_neighbors):
        row = self.graph[node]
        neighbors = row.nonzero()[1]
        links = row.data
        if len(links) == 0:
            pass
        random_choice = np.random.choice(len(links), size=num_neighbors)
        return neighbors[random_choice], links[random_choice]

    def take_one_step(self, nodes):
        neighbors, links = [], []
        for node in nodes:
            neighbor, link = self.get_neighbors(node, 1)
            neighbors.append(neighbor[0])
            links.append(link[0])
        return neighbors, links

    def random_walk(self, node, num_walkers, max_len, end_type):
        neighbors, links = self.get_neighbors(node, num_walkers)
        batch, paths, nodes = [], [], []
        for i, neighbor in enumerate(neighbors):
            link = np.random.choice(self.id_to_links[links[i]])
            if self.node_to_type[neighbor]==end_type:
                batch.append([node, link, neighbor])
            else:
                paths.append([node, link, neighbor])
                nodes.append(neighbor)
        for i in range(1, max_len+1):
            neighbors, links = self.take_one_step(nodes)
            next_nodes, new_paths = [], []
            for i, neighbor in enumerate(neighbors):
                link = np.random.choice(self.id_to_links[links[i]])
                if self.node_to_type[neighbor] == end_type:
                    batch.append(paths[i]+[link, neighbor])
                else:
                    new_paths.append(paths[i]+[link, neighbor])
                    next_nodes.append(neighbor)
            if next_nodes:
                nodes = next_nodes
                paths = new_paths
            else:
                break
        return batch

    def get_neighbors_by_link(self, node, link_type):
        if node in self.neighbor_by_link_dict:
            if link_type in self.neighbor_by_link_dict[node]:
                neighbors = self.neighbor_by_link_dict[node][link_type]
                if len(neighbors) > self.data_threshold:
                    r = np.random.randint(len(neighbors) - self.data_threshold)
                    neighbors = neighbors[r:r + self.data_threshold]
                return neighbors
        return []

    def random_walk_with_pattern(self, src_node, pattern):
        nodes = self.get_neighbors_by_link(src_node, pattern[0])
        if len(nodes) == 0:
            return []
        for link in pattern[1:]:
            new_nodes = []
            for i, node in enumerate(nodes):
                next_nodes = self.get_neighbors_by_link(node, link)
                if len(next_nodes) == 0:
                    continue
                if len(next_nodes) > self.data_threshold:
                    next_nodes = np.random.choice(next_nodes, size=self.data_threshold)
                new_nodes += next_nodes

            if len(new_nodes) == 0:
                return []
            else:
                nodes = new_nodes
        return [[node, src_node] for node in nodes]

    def init_pattern(self, nodes, num_pattern, num_walkers, max_len, end_type, reverse_path=False, verbose=False):
        patterns = []
        if num_pattern is None:
            for node in tqdm(nodes, disable=not verbose):
                paths = self.random_walk(node, num_walkers, max_len, end_type)
                patterns += paths
        else:
            while len(patterns) < num_pattern:
                for node in tqdm(nodes, disable=not verbose):
                    paths = self.random_walk(node, num_walkers, max_len, end_type)
                    patterns += paths
                    if len(patterns) >= num_pattern:
                        break
            patterns = patterns[:num_pattern]
        # pattern do not have to reverse
        if reverse_path:
            patterns = np.array([np.array(path[::-1][1::2]) for path in patterns])
        else:
            patterns = np.array([np.array(path[1::2]) for path in patterns])

        new_patterns = map(tuple, patterns)
        pattern_count = Counter(new_patterns)
        patterns, counts = [], []
        for pattern in pattern_count:
            patterns.append(pattern)
            counts.append(pattern_count[pattern])
        counts = [float(i) / sum(counts) for i in counts]

        self.counts = counts
        self.patterns = patterns

    def sample_pattern(self, n):
        return np.random.choice(self.patterns, size=n, p=self.counts)

    def collect_data(self, num_data):
        self.pattern = self.sample_pattern(1)[0]

        # path only contain nodes
        # it is faster if num_walker is large enough so that following while loop only run once
        batch = []
        shuffle(self.train_nodes)
        buget = 10
        while True:
            for node in self.train_nodes:
                paths = self.random_walk_with_pattern(node, self.pattern)
                batch += paths
                if len(batch) >= num_data:
                    return np.array(batch), [str(edge) for edge in self.pattern[::-1]]
            buget -= 1
            if buget == 0:
                self.pattern = self.sample_pattern(1)[0]
                buget = 10

    def save_pattern(self, path):
        patterns = self.patterns
        patterns = [tuple([int(i) for i in pattern]) for pattern in patterns]
        json.dump({'patterns':patterns, 'counts':self.counts}, open(path, 'w'),  indent=2)

    def load_pattern(self, path):
        data = json.load(open(path, 'r'))
        self.patterns = data['patterns']
        self.counts = data['counts']