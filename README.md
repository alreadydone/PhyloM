
# Noise Inference

# Branching Inference

# Noise Elimination


```console
foo@bar:~$ python PhyloR.py --restore_from --nCells --nMuts
--nTestMats --fp --fn --output_dir --batch_size --input_dimension
--betta --ms_dir
```


## System configs
All the experiments in this work are performed using **Carbonate**, a computer cluster at Indiana University. We used deep learning (DL) nodes in Carbonate.
These nodes features 12 GPU-accelerated Lenovo ThinkSystem SD530 , each equipped with two Intel Xeon Gold 6126 12-core CPUs, two NVIDIA GPU accelerators (eight with Tesla P100s; four with Tesla V100s), and 192 GB of RAM. A Red Hat Enterprise 7.x is running on these nodes and we used Python 3.6.6 :: Anaconda, Inc., Tensorflow 1.8.0, and Keras 2.2.0.
