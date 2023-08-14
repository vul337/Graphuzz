import dgl
import os
import torch
from filehash import FileHash
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import pickle
import tqdm
import random

class CDGDataset(DGLDataset):
    def __init__(self, name, raw_dir=None, force_reload = False, verbose = False, add_self_loop=True, num_labels= 1, random_attr_idx = None):
        self._name = name
        if raw_dir is None:
            raise FileNotFoundError
        else:
            self._raw_dir = raw_dir

        self.graph_file_path = []
        self._labels = []
        self.labels = None
        self._len = 0
        self._num_labels = 1
        self.random_attr_idx = random_attr_idx
        super(CDGDataset, self).__init__(name=name,
                                raw_dir=raw_dir,
                                save_dir=raw_dir,
                                force_reload=force_reload,
                                verbose=verbose)
    
    def process(self):
        if not os.access(self.raw_dir, os.F_OK):
            raise FileNotFoundError
        for file in tqdm.tqdm(os.listdir(self.raw_dir)):
            #if len(self.graph_file_path) % 100 == 0:
            #    print("[+] load {} graphs".format(len(self.graph_file_path)))
            if not file.endswith(".dgl"):
                continue
            file_path = os.path.join(self.raw_dir, file)
            label = float(file.split("_")[-1][:-4]) * 100
            if label == float('inf'):
                print("bad data:", file_path)
                continue
            #self.graph_list.append(dgl.add_self_loop(dgl.load_graphs(file_path)[0][0]))
            self.graph_file_path.append(file_path)
            self._labels.append(label)
        
        combined = list(zip(self.graph_file_path, self._labels))
        random.shuffle(combined)
        self.graph_file_path, self._labels = zip(*combined)

        self.labels = torch.tensor(self._labels, dtype=torch.float32)
        self._len = len(self.graph_file_path)
        for i in range(0, self._len):
            if i >= self._len:
                break
            if self.labels[i].item() == float('inf') or self.labels[i].item() != self.labels[i].item():
                self._len -= 1
                print(self.graph_file_path[i])
                self._labels.pop(i)
                self.graph_file_path.pop(i)
        self.labels = torch.tensor(self._labels, dtype=torch.float32)
    
    def download(self):
        return False

    def save(self):
        cache_path = os.path.join(self.raw_dir, "cache.pkl")
        with open(cache_path, "wb") as pf:
            pickle.dump([self.graph_file_path, self.labels], pf)
        #dgl.save_graphs(graph_path, self.graph_list, {"labels": self.labels})
    
    def load(self):
        cache_path = os.path.join(self.raw_dir, "cache.pkl")
        with open(cache_path, "rb") as pf:
            self.graph_file_path, self.labels = pickle.load(pf)
        self._len = len(self.graph_file_path)
    
    def has_cache(self):
        graph_path = os.path.join(self.raw_dir, "cache.pkl")
        if os.path.exists(graph_path):
            return True

    def __getitem__(self, index):
        if index >= len(self.graph_file_path):
            raise IndexError
        g = dgl.load_graphs(self.graph_file_path[index])[0][0]
        if self.random_attr_idx != None:
            for i in range(0, len(g.ndata["attr"])):
                g.ndata["attr"][i][self.random_attr_idx] = random.randint(0, 100)
        return dgl.add_self_loop(g), self.labels[index]
    
    def __len__(self):
        return len(self.graph_file_path)

    def statistics(self):
        return self.__getitem__(0)[0].ndata["attr"].shape[1], self._num_labels


class DoubleGraphDataSet(DGLDataset):
    def __init__(self, name, raw_dir=None, add_self_loop=True, num_labels= 7):
        self._name = name
        if raw_dir is None:
            raise FileNotFoundError
        else:
            self._raw_dir = raw_dir

        self._load()
    
    def load(self):
        md5hasher = FileHash("md5")
        for dirpath, dirnames, filenames in os.walk(self._raw_dir):
            for filename in filenames:
                if filename.endswith(".dglfull"):
                    _hash = md5hasher.hash_file(os.path.join(dirpath, filename))
                    if _hash in self._graph_full:
                        continue
                    else:
                        glist, label_dict = dgl.load_graphs(os.path.join(dirpath, filename), [0])
                        if self._add_self_loop:
                            self._graph_full[_hash] = dgl.add_self_loop(glist[0])
                        else:
                            self._graph_full[_hash] = glist[0]
        for dirpath, dirnames, filenames in os.walk(self._raw_dir):
            for filename in filenames:
                if filename.endswith(".dglsub"):
                    _sub_info = filename[:filename.find(".dglsub")].split("_")
                    _full_graph_hash = _sub_info[0]
                    _sub_graph_index = _sub_info[1]
                    _sub_graph_label = int(_sub_info[2].strip())
                    if _hash not in self._graph_full:
                        raise FileNotFoundError("full graph not found!")
                    else:
                        _glist, label_dict = dgl.load_graphs(os.path.join(dirpath, filename), [0])
                        if _full_graph_hash not in self._graph_sub:
                            self._graph_sub[_full_graph_hash] = []
                        if self._add_self_loop:
                            self._graph_sub[_full_graph_hash].append((dgl.add_self_loop(_glist[0]), _sub_graph_label))
                        else:
                            self._graph_sub[_full_graph_hash].append((_glist[0], _sub_graph_label))
                        self._len += 1
        idx = 0
        for _full_hash in self._graph_sub.keys():
            _sub_idx = 0
            for _sub_graph in self._graph_sub[_full_hash]:
                self._idxtb[idx] = (_full_hash, _sub_idx)
                _sub_idx += 1
                idx += 1

    def _load(self):
        try:
            self.load()
        except KeyboardInterrupt:
            raise

    def __getitem__(self, index):
        if index >= self._len:
            raise IndexError
        full_graph = self._graph_full[self._idxtb[index][0]]
        sub_graph = self._graph_sub[self._idxtb[index][0]][self._idxtb[index][1]]
        return dgl.batch([full_graph, sub_graph[0]]), sub_graph[1]
    
    def __len__(self):
        return self._len

    def statistics(self):
        return self.__getitem__(0)[0].ndata["attr"].shape[1], self._num_labels