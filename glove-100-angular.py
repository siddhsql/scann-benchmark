# https://medium.com/@DataPlayer/scalable-approximate-nearest-neighbour-search-using-googles-scann-and-facebook-s-faiss-3e84df25ba
import argparse
import numpy as np
import h5py
import os
import time

import scann

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="input file", type=str, required=True)
parser.add_argument("--output", help="output file", type=str, required=True)
parser.add_argument("--threads", help="number of threads", type=int, default=4)
parser.add_argument("-K", type=int, default=10)

args = parser.parse_args()

if os.path.exists(args.output):
    raise RuntimeError(f"The file '{args.output}' exists.")

if not os.path.exists(args.input):
    raise RuntimeError(f"The file '{args.input}' does not exist.")

with h5py.File(args.input, "r") as f:
    train = f["train"]
    test = f["test"]
    dim = len(train[0])
    max_elements = len(train)
	
    normalized_dataset = train / np.linalg.norm(train, axis=1)[:, np.newaxis]
# configure ScaNN as a tree - asymmetric hash hybrid with reordering
# anisotropic quantization as described in the paper; see README

# use scann.scann_ops.build() to instead create a TensorFlow-compatible searcher
# https://github.com/google-research/google-research/blob/master/scann/scann/scann_ops/py/scann_ops_pybind.py
    print(f"Builing index...");
    start_time = time.time()
# 10 is K - the number of nearest neighbors to search for. we need to set this in advance. this sucks! and maybe the reason chroma uses hnswlib instead of scaNN
    searcher = scann.scann_ops_pybind.builder(normalized_dataset, 10, "dot_product") \
    .tree(num_leaves=2000, \
            num_leaves_to_search=100, \
            training_sample_size=250000) \
    .score_ah( \
    		2, \
		anisotropic_quantization_threshold=0.2) \
    .reorder(100) \
    .build()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time: {elapsed_time} seconds.")
	
	# this will search the top 100 of the 2000 leaves, and compute
	# the exact dot products of the top 100 candidates from asymmetric
	# hashing to get the final top 10 candidates.
    print("Querying index...");
    start_time = time.time()
	# https://github.com/google-research/google-research/blob/master/scann/scann/scann_ops/py/scann_ops_pybind.py#L60
    neighbors, distances = searcher.search_batched(test)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time: {elapsed_time} seconds.")

    print("saving results...")
    with h5py.File(args.output, "w") as fout:
        fout.create_dataset("labels", data=neighbors, dtype="i4")
        fout.create_dataset("distances", data=distances, dtype="f4")

    # https://stackoverflow.com/a/68705476/147530
    print("saving index...")
    INDEX_DIR = './index'
    os.makedirs(INDEX_DIR, exist_ok=False)
    searcher.serialize(INDEX_DIR)

print("done")

