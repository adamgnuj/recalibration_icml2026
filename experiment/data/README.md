# Datasets
Run all cells in 
- [gen_sinus.ipynb](base_datasets/gen_sinus.ipynb)
- [get_bike.ipynb](base_datasets/get_bike.ipynb)
- [prepare_uci.ipynb](uci_datasets/prepare_uci.ipynb)

to prepare the exported splits for later use.

We are using the [repository](https://github.com/yaringal/DropoutUncertaintyExps) of the paper
["Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning" (2015)](http://www.cs.ox.ac.uk/people/yarin.gal/website/publications.html#Gal2015Dropout),
to obtain the data and pre-defined splits for the UCI regression benchmark.
The cloning of their repository is done in the notebook [prepare_uci.ipynb](uci_datasets/prepare_uci.ipynb) automatically.

The file `data_split_iterator.py` is used to iterate over all datasets and splits in the subdirectories of this directory.