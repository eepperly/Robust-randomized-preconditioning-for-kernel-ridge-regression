#!/usr/bin/env python3

import requests
from libsvm.svmutil import svm_read_problem
from scipy.io import savemat 
from tqdm import tqdm
import bz2
import lzma
import os
# Auxiliary functions

def download_file(url, directory, compression = None):
    print(f"Downloading {url}")
    filename = url.split('/')[-1]
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    filepath = directory + filename
    with open(filepath, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
        file.close()    
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")    
    return decompress_file(filepath)  

def decompress_file(filepath):
    
    def decompress_with(libr, extension):
        newfilepath = filepath[:(-len(extension))]
        with libr.open(filepath) as f, open(newfilepath, 'wb') as fout:
            file_content = f.read()
            fout.write(file_content)
            os.remove(filepath)
        return newfilepath
    
    if filepath.lower().endswith("bz2"):
        return decompress_with(bz2, ".bz2")
    elif filepath.lower().endswith("xs"):
        return decompress_with(lzma, ".xs")
    else:
        return filepath
            

raw_data_directory = "../data/raw/"
mat_data_directory = "../data/preprocessed/"

# If a dataset includes an url for testing, we use the two datasets, otherwise we split training according
# to ntr (number of training datapoints) and nts (number of test datapoints).
datasets = {
    # "HIGGS": {
    #     "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/HIGGS.xz",
    #     "ntr": 10500000,
    #     "nts": 500000,
    # },
    "a9a": {
        "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a",
        "test": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t",
    },
    "cod-rna": {
        "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna",
        "test": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna.t",
    },
    "covtype.binary": {
        "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2",
        "ntr" : 481012,
        "nts" : 100000,
    },
    "w8a": {
        "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a",
        "test": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a.t"
    },
    "ijcnn1": {
        "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.bz2",
        "test": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.t.bz2",
    },
        "skin_nonskin" : {
        "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/skin_nonskin",
        "ntr": 196045,
        "nts": 49012,
    },
    "australian": {
        "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/australian",
        "ntr": 590,
        "nts": 100,
    }
}

datasets = {
    # "connect-4": {
    #     "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/connect-4",
    #     "ntr": 54045,
    #     "nts": 13512,
    # },
    # "sensorless": {
    #     "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/Sensorless.scale",
    #     "test": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/Sensorless.scale.tr",
    # },
    "sensit_vehicle" : {
        "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/combined.bz2",
        "test": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/combined.t.bz2",
    },
     "SUSY" : {
        # "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_unigram.svm.xz",
        "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/SUSY.xz",
        "ntr": 45000,
        "nts": 5000,
    },
}

large_datasets = {
    "epsilon" : {
        "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2",
        "test" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.bz2",
    },
    "SUSY" : {
        # "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_unigram.svm.xz",
        "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/SUSY.xz",
        "ntr": 45000,
        "nts": 5000,
    },
    "YearPredictionMSD" : {
        "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.bz2",
        "test" : "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.t.bz2",
    },
}
for k, data in datasets.items():
    url = data["train"]
    dataset_path = download_file(url, raw_data_directory)
    # r = requests.get(url, allow_redirects=True)
    # filename = url.split('/')[-1]
    # dataset_path = raw_data_directory + filename
    # open(dataset_path, 'wb').write(r.content)
    Ytr, Xtr = svm_read_problem(dataset_path, return_scipy = True)
    if "test" in data:
        url_ts = data["test"]
        dataset_path_ts = download_file(url_ts, raw_data_directory)
        # filename_ts = url_ts.split('/')[-1]
        # r_ts = requests.get(url_ts, allow_redirects=True)
        # dataset_path_ts = raw_data_directory + filename_ts
        # open(dataset_path_ts, 'wb').write(r_ts.content)
        Yts, Xts = svm_read_problem(dataset_path_ts, return_scipy = True)
    else:
        Ytraux = Ytr[:(data["ntr"])]
        Xtraux = Xtr[:(data["ntr"])]
        Yts = Ytr[data["ntr"]:(data["ntr"] + data["nts"])]
        Xts = Xtr[data["ntr"]:(data["ntr"] + data["nts"])]
        Ytr = Ytraux
        Xtr = Xtraux
        
    datadir = {'Xtr': Xtr , 'Ytr': Ytr, 'Xts': Xts, 'Yts': Yts}
    savemat(mat_data_directory + k +".mat", datadir)
    
    
    
