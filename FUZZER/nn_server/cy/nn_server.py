import os
import socket
import sys
import threading
import mmap
import ctypes
import random
import torch
import pickle
from exportTrainingSet import CDFG
import dgl
import numpy as np
import math
import time
import zlib
import copy
from torch.cuda.amp import autocast as autocast
#import pyroscope
#
##cudnn.benchmark = True
#
#pyroscope.configure(
#  application_name = "dcfuzz.fuzzer.nn_server", # replace this with some name for your application
#  server_address   = "http://172.17.0.1:4040", # replace this with the address of your pyroscope server
#)

import re
import subprocess

def execute_command(cmd):
    cmd_args = cmd.split()
    out = subprocess.check_output(cmd_args)
    return out.decode('utf-8')

def find_availabel_gpu():
    cmd = "nvidia-smi --query-gpu=memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits"
    gpus_info = execute_command(cmd)
    gpus_info = gpus_info.strip().split("\n")
    gpus_info = [[int(y.lstrip().strip()) for y in x.split(',')] for x in gpus_info]
    for i in range(0, len(gpus_info)):
        gpus_info[i].append(i)
    return sorted(gpus_info, key=lambda x:x[1], reverse=True)[0][-1]


class NNModel():
    def __init__(self, 
        device = "cuda:{}".format(find_availabel_gpu()), 
        model_path=os.path.join(__file__[:__file__.rfind('/')], "../", "model.pth"),
        cdfg_path = os.path.join(__file__[:__file__.rfind('/')], "../", "CDFG.pkl")
    ):
        self.device = torch.device(device)
        self.model = torch.load(model_path, map_location=device).to(device)
        #self.model = self.model.half()
        torch.compile(self.model)

        with open(cdfg_path, 'rb') as cdfg_file:
            self.cdfg = pickle.load(cdfg_file)

    def preprocess_bits_to_graph(self, virgin_bits, seed_bits, virgin_changed=True):
        return dgl.add_self_loop(self.cdfg.exportGraph(virgin_bits, seed_bits, virgin_changed=virgin_changed))

    def predict(self, virgin_bits, seed_bits):
        dgl_graph = self.preprocess_bits_to_graph(virgin_bits, seed_bits)
        self.model.eval()
        out, _ = self.model(dgl_graph.to(self.device))
        return out

    def predict_batch(self, batch):
        self.model.eval()
        with autocast(dtype=torch.float16):
            out, perms = self.model(batch.to(self.device))
        return out, perms

class SeedQueue():
    def __init__(self):
        self.seeds = list()
    
    def append(self, bitmap):
        item = [bitmap, time.time(), None, -99999999.0, True]
        self.seeds.append(item)
    
    def searchIndexsNeedUpdate(self, time_interval):
        '''
        time_interval: seconds
        '''
        indexes = []
        for i in range(0, len(self.seeds)):
            if (time.time() - self.seeds[i][1] > time_interval) or type(self.seeds[i][2]) != np.ndarray:
                if self.seeds[i][4] != False:
                    indexes.append(i)
        return indexes
    
    def fetchBitmap(self, index):
        res = zlib.decompress(self.seeds[index][0])
        #print(hex(len(res)))
        return res

    def getSize(self):
        return len(self.seeds)
    
    def clearSeeds(self):
        self.seeds.clear()
    
    def updateSeed(self, idx, weight, attention_bits):
        self.seeds[idx][1] = time.time()
        self.seeds[idx][2] = attention_bits
        self.seeds[idx][3] = weight

    def fetchWeight(self, index):
        return self.seeds[index][3]

    def fetchAttention(self, index):
        if type(self.seeds[index][2]) == type(None):
            return None
        else:
            return zlib.compress(self.seeds[index][2].tobytes())

    def updateBitmap(self, index, bitmap):
        self.seeds[index][0] = bitmap
        self.seeds[index][4] = True
    
    def updateEnable(self, index, enable):
        self.seeds[index][4] = enable


class NNServer():
    def __init__(self, nn_model: NNModel, server_path = "/tmp/nn_server.sock"):
        self.server_path = server_path
        try:
            os.unlink(server_path)
        except OSError:
            if os.path.exists(server_path):
                raise
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(server_path)
        server.listen(1)
        self.server = server
        self.threads = []
        self.clients = []
        self.mems = []
        self.fds = []
        self.lock = threading.Lock()
        self.seed_queue_lock = threading.Lock()
        self.seed_queue = SeedQueue()
        self.nn_model = nn_model

        self.seed_thread = threading.Thread(target = self.seed_evaluator)
        self.seed_thread.daemon = True
        self.seed_thread.start()
        self.map_size = None
    
    def seed_evaluator(self):
        attention_matrix = None
        id_bb_dict = np.array(self.nn_model.cdfg.id_bb_dict, dtype="int")
        while True:
            try:
                mem_obj = self.mems[-1]
                break
            except:
                time.sleep(5)
        while True:
            time.sleep(60)
            self.seed_queue_lock.acquire()
            batch_update = []
            idxs_need_update = self.seed_queue.searchIndexsNeedUpdate(30 * 60)
            #print(idxs_need_update)
            #self.virgin_changed = True
            virgin_changed = self.virgin_changed
            self.seed_queue_lock.release()
            virgin_bits = copy.deepcopy(self.virgin_bits)
            for idx in idxs_need_update:
                try:
                    batch_update.append(
                        (
                            idx,
                            self.nn_model.preprocess_bits_to_graph(
                                virgin_bits, 
                                self.seed_queue.fetchBitmap(idx), 
                                virgin_changed = virgin_changed
                            )
                        )
                    )
                    virgin_changed = False
                except:
                    __import__("traceback").print_exc()
                    self.seed_queue.updateEnable(idx, False)                        
                if (len(batch_update) >= 32 or idx == idxs_need_update[-1]) and len(batch_update) > 0:
                    batch_graph_update = dgl.batch([x[1] for x in batch_update])
                    #print(batch_graph_update)
                    try:
                        batch_reverse_indexes = [[] for i in range(0, batch_graph_update.batch_size)]
                        weights, perms = self.nn_model.predict_batch(batch_graph_update)
                        weights = weights.detach().to("cpu")
                        perms = perms.detach().to("cpu")
                        pool_ratio = self.nn_model.model.convpools[0].pool.ratio
                        pool_layers = self.nn_model.model.num_convpools
                        X = batch_graph_update.num_nodes() // batch_graph_update.batch_size
                        Xs = []
                        Xs.append(math.ceil(X * pool_ratio))
                        for i in range(1, pool_layers):
                            Xs.append(math.ceil(Xs[i-1] * pool_ratio))
                        #print(Xs)
                        batch_indexes = [[] for i in range(0, batch_graph_update.batch_size)]
                        for i in range(0, batch_graph_update.batch_size):
                            for j in range(0, pool_layers):
                                batch_indexes[i].append(
                                    torch.tensor(
                                        perms[
                                            sum(Xs[:j]) * batch_graph_update.batch_size + i * Xs[j] : 
                                            sum(Xs[:j]) * batch_graph_update.batch_size + (i + 1) * Xs[j]
                                        ],
                                        dtype=torch.long
                                    ).to("cpu")
                                )
                        for i in range(0, batch_graph_update.batch_size):
                            batch_reverse_indexes[i].append(
                                torch.tensor(batch_indexes[i][0] - X * i, dtype=torch.long).to("cpu")
                            )
                            for j in range(1, pool_layers):
                                #print(pool_ratio, pool_layers, len(batch_reverse_indexes[i][j-1]), max(batch_indexes[i][j] - Xs[j] * i))
                                batch_reverse_indexes[i].append(
                                    torch.tensor(
                                        batch_reverse_indexes[i][j-1][batch_indexes[i][j] - Xs[j-1] * i],
                                        dtype=torch.long
                                    ).to("cpu")
                                )
                        print(len(batch_update), len(weights), weights)
                    except:
                        __import__("traceback").print_exc()
                        print(batch_graph_update)
                        weights = [100.0 for i in range(0, len(batch_update))]
                        #print(len(batch), len(weights), weights)
                    for i in range(0, len(weights)):
                        weight = weights[i]
                        #print(len(batch_reverse_indexes[i]))
                        batch_reverse_indexes[i] = torch.cat(batch_reverse_indexes[i], dim=-1).to("cpu")
                        unique, unique_counts = np.unique(batch_reverse_indexes[i], return_counts=True)
                        unique = np.array(unique, dtype=np.int32)
                        unique_counts = np.array(unique_counts, dtype=np.uint8)
                        try:
                            attention_matrix = np.zeros(65536 if self.map_size==None else self.map_size, dtype=np.uint8)
                            attention_matrix[id_bb_dict[unique]] = unique_counts
                            self.seed_queue.updateSeed(batch_update[i][0], weight, attention_matrix)
                        except:
                            __import__("traceback").print_exc()
                            attention_matrix = np.zeros(65536 if self.map_size==None else self.map_size, dtype=np.uint8)
                            self.seed_queue.updateSeed(batch_update[i][0], weight, attention_matrix)
                    batch_update = []

    def thread_handler(self, *args):
        print("thread handler spwning! ...")
        clientsocket, address, self.mem_obj = args
        mem_obj = self.mem_obj
        #clientsocket.settimeout(5)
        self.virgin_bits = None
        self.virgin_changed = True
        while True:
            try:
                cmd = clientsocket.recv(8)
                #if len(cmd) > 0:
                #    print(cmd)
                
                if cmd == b'updt map':
                    clientsocket.send(b'OK')
                    idx = int.from_bytes(clientsocket.recv(8), byteorder="little", signed=False)
                    clientsocket.send(b'OK')
                    length = int.from_bytes(clientsocket.recv(8), byteorder="little", signed=False)
                    seed_bits = mem_obj.read(length)
                    mem_obj.seek(0)
                    self.seed_queue_lock.acquire()
                    self.seed_queue.updateBitmap(idx, seed_bits)
                    self.seed_queue_lock.release()
                    if self.map_size == None:
                        self.map_size = len(self.seed_queue.fetchBitmap(0)) << 3
                        print("MAP SIZE:", hex(self.map_size))
                    clientsocket.send(b'OK')

                if cmd == b'add seed':
                    clientsocket.send(b'OK')
                    length = int.from_bytes(clientsocket.recv(8), byteorder="little", signed=False)
                    seed_bits = mem_obj.read(length)
                    mem_obj.seek(0)
                    self.seed_queue_lock.acquire()
                    self.seed_queue.append(seed_bits)
                    self.seed_queue_lock.release()
                    clientsocket.send(b'OK')

                if cmd == b'chck idx':
                    clientsocket.send(bytes(ctypes.c_uint64(self.seed_queue.getSize())))
                
                if cmd == b'clr seed':
                    self.seed_queue.clearSeeds()
                    clientsocket.send(b'OK')

                elif cmd == b'virg now':
                    #print("[*] virg now")
                    if self.map_size == None:
                        self.virgin_bits = mem_obj.read(mem_obj.size())
                    else:
                        self.virgin_bits = mem_obj.read(self.map_size)
                    clientsocket.send(b'OK')
                    mem_obj.seek(0)
                    self.virgin_changed = True

                elif cmd == b'close':
                    print(address, " disconnected")
                    break
                
                elif cmd == b'get weit':
                    clientsocket.send(b'OK')
                    idx = int.from_bytes(clientsocket.recv(8), byteorder="little", signed=False)
                    #self.seed_queue_lock.acquire()
                    clientsocket.send(bytes(ctypes.c_double(self.seed_queue.fetchWeight(idx))))
                    #self.seed_queue_lock.release()
                    if clientsocket.recv(5) == b'ATTEN':
                        mem_obj.seek(0)
                        #self.seed_queue_lock.acquire()
                        attention_bits = self.seed_queue.fetchAttention(idx)
                        if type(attention_bits) != bytes:
                            attention_bits = zlib.compress(np.zeros(self.map_size, dtype=np.uint8).tobytes())
                        mem_obj.write(attention_bits)
                        #self.seed_queue_lock.release()
                        mem_obj.seek(0)
                        clientsocket.send(bytes(ctypes.c_uint64(len(attention_bits))))

            except Exception as e:
                __import__("traceback").print_exc()
                print(address, " error")
                break

        # clean resource
        clientsocket.shutdown(socket.SHUT_RDWR)
        clientsocket.close()
        self.lock.acquire()
        idx = 0
        while True:
            if idx >= len(self.clients):
                break
            if clientsocket == self.clients[idx]:
                self.clients.pop(idx)
                os.close(self.fds[idx])
                self.fds.pop(idx)
                self.mems[idx].close()
                self.mems.pop(idx)
                break
            idx += 1
        self.lock.release()

    def main_loop(self):
        print("[*] listening ...")
        while True:
            clientsocket, address = self.server.accept()
            for i in range(0, len(self.threads)):
                self.threads[i].join(0.1)
            self.lock.acquire()
            self.clients.append(clientsocket)
            print("{} connected".format(address))
            new_memfd = clientsocket.recv(1025)
            print("new memfd: {}".format(new_memfd))
            fd = os.open(new_memfd, os.O_RDWR)
            if fd < 0:
                self.clients[-1].close()
                self.clients.pop()
                continue
            self.fds.append(fd)
            mem_obj = mmap.mmap(fd, length = 0, prot=mmap.PROT_READ | mmap.PROT_WRITE, flags=mmap.MAP_SHARED)
            self.mems.append(mem_obj)
            clientsocket.send(b'OK')
            thread = threading.Thread(target = self.thread_handler, args = (clientsocket, address, mem_obj))
            self.threads.append(thread)
            thread.daemon = True
            thread.start()
            self.lock.release()

if __name__ == "__main__":
    nn_model = NNModel()
    nn_server = NNServer(nn_model)
    nn_server.main_loop()
