# NEP
---------------

Code for the paper " Neural Embedding Propagation on Heterogeneous Networks" ICDM 2019

### Required Inputs
---------------
see the **data.zp**


### Command
---------------

To train NEP model with default setting, please first unzip **data.zip** and then run
```
python3 nep/main.py
```

You can specify the parameters. The variables names are self-explained.


### Key Parameters
---------------

**dataset**: the path to data folder should be ./data/**dataset**.

**target_node_type**: should be consistent with the node type in **node.dat** file, in example dblp-sub dataset, the target node type is 'a'.

**train_ratio**: how many of labeled data to be used as train data.

**superv_ratio**: how many of train data to be exposed to model (used in experiment of label efficiency).

**path_max_length**: the maximum length of a pattern (used in the experiment of path length).

## Citation
---------------

Please cite the following two papers if you are using our tool. Thanks!


