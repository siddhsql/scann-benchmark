# README.md

Give scann algorithm a test drive. Measure its perfromance and compare to `hnswlib`.

[docs](https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md), [pypi](https://pypi.org/project/scann/)

The implementation is designed for x86 processors with AVX2 support. so it doesn't run on mac. includes search space pruning and quantization for Maximum Inner Product Search and also supports other distance functions such as Euclidean distance.

Limitations:

- The number of neighbors to search has to be specified a-priori and you are stuck with it
- You also have to declare the number of samples in training set in advance

## Performance

```
.env) ubuntu@104-171-202-206:~/morpheus/benchmarks/scann$ python glove-100-angular.py  --input /home/ubuntu/morpheus/benchmarks/data/glove-100-angular.hdf5 --output scann-output.hdf5
Builing index...
[libprotobuf WARNING external/com_google_protobuf/src/google/protobuf/text_format.cc:339] Warning parsing text-format research_scann.ScannConfig: 43:11: text format contains deprecated field "min_cluster_size"
2023-07-12 00:56:46.400361: I scann/partitioning/partitioner_factory_base.cc:71] Size of sampled dataset for training partition: 249797
2023-07-12 00:56:49.440455: I ./scann/partitioning/kmeans_tree_partitioner_utils.h:102] PartitionerFactory ran in 3.039984134s.
Time: 12.717264890670776 seconds.
Querying index...
Time: 1.7619037628173828 seconds.
saving results...
saving index...
done
```

## Post-mortem

```
>>> import h5py
>>> f = h5py.File("glove-100-output.hdf5", "r")
>>> labels = f["labels"]
>>> distances = f["distances"]
>>> labels[0]
array([  97478,  846101,  671078,  727732,  544474, 1133489,  723915,
        660281,  566421, 1093917], dtype=int32)
>>> distances[0:10]
array([[2.5518737, 2.539792 , 2.5383418, 2.5097368, 2.4656374, 2.4636059,
        2.4552207, 2.4540553, 2.452435 , 2.4514375],
       [3.2823462, 3.2309842, 3.139618 , 3.138874 , 3.1332953, 3.104702 ,
        3.0887508, 3.086329 , 3.078837 , 3.0745459],
       [3.6476276, 3.5127845, 3.4592834, 3.447577 , 3.4031463, 3.3718703,
        3.3314686, 3.316904 , 3.2964623, 3.2893581],
       [4.204627 , 4.0689836, 4.066025 , 3.9838798, 3.96854  , 3.9577572,
        3.949725 , 3.9322834, 3.894211 , 3.8709254],
       [3.9441233, 3.925972 , 3.8196406, 3.8000064, 3.781902 , 3.7655559,
        3.7611575, 3.7545211, 3.725743 , 3.7161531],
       [3.694106 , 3.6001906, 3.5903373, 3.5724015, 3.56171  , 3.5421972,
        3.4934688, 3.4710512, 3.4614425, 3.4374561],
       [3.5119357, 3.438562 , 3.386167 , 3.3124828, 3.2715507, 3.1537023,
        3.1524181, 3.0781093, 3.0664256, 3.0467188],
       [3.0952096, 3.0838299, 3.0580523, 3.0467434, 3.045415 , 3.0435882,
        3.0385222, 3.0022588, 2.9921527, 2.9860868],
       [4.2097073, 3.6695375, 3.4795332, 3.2583075, 3.1545815, 3.121883 ,
        3.1047714, 3.0916016, 3.0421665, 3.0347936],
       [3.3354912, 3.1681032, 3.1615405, 3.123613 , 3.1049838, 3.0908027,
        3.0715985, 3.0576684, 3.0277605, 3.021196 ]], dtype=float32)
```
