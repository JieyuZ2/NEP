## Implementation of *NEP*, ICDM 2019.

Please cite the following work if you find the code useful.

```
@inproceedings{yang2019neural,
	Author = {Yang, Carl and Zhang, Jieyu and Han, Jiawei},
	Booktitle = {ICDM},
	Title = {Neural embedding propagation on heterogeneous networks},
	Year = {2019}
}
```
Contact: Carl Yang (yangji9181@gmail.com)

### Inputs
---------------
See **data.zip** as am example.


### Command
---------------

To train NEP model with default setting, please first unzip **data.zip** and then run
```
python3 nep/main.py
```

You can specify the parameters. The variable names are self-explaining.


### Key Parameters
---------------

**dataset**: the path to data folder should be ./data/**dataset**.

**target_node_type**: should be consistent with the node type in **node.dat** file, in example dblp-sub dataset, the target node type is 'a'.

**train_ratio**: how many of labeled data to be used as train data.

**superv_ratio**: how many of train data to be exposed to model (used in experiment of label efficiency).

**path_max_length**: the maximum length of a pattern (used in the experiment of path length).
