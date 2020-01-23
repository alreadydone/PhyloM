
# Getting started

You can get the code for different parts of this project using the following command.

```console
git clone https://github.com/MHEAAA/PhyloM.git
```


# Noise Elimination
The code for this part could be found under RL folder. We provided the **Keras** weights for 10x10 size matrices under folder RL/Model. Using the following command a model based on the specified number of cells and mutations will be compiled, and provided weights will be imported to the model. The program will generate a number of evaluation instances and provide the solution it can find.

```console
foo@bar:~$ python PhyloR.py Options

Required:

--nCells         Number of rows of input matrices (Number of cells)
--nMuts          Number of columns of input matrices (Number of mutations)
--fp             False positive rate
--fn             False negative rate
--nTestMats      Number of test matrices
--output_dir     Output directory for storing solution matrices
--restore_from   Directory to restore the model and weights from
--ms_dir         MS program directory

Optional:

--gamma          Cost function hyper-parameter
```

# Noise Inference
The code for this part has been provided under NE folder.

# Branching Inference
The code for this part is placed under BI folder.

## System configs
All the experiments in this work are performed using **Carbonate**, a computer cluster at Indiana University. We used deep learning (DL) nodes in Carbonate.
These nodes features 12 GPU-accelerated Lenovo ThinkSystem SD530 , each equipped with two Intel Xeon Gold 6126 12-core CPUs, two NVIDIA GPU accelerators (eight with Tesla P100s; four with Tesla V100s), and 192 GB of RAM. A Red Hat Enterprise 7.x is running on these nodes and we used Python 3.6.6 :: Anaconda, Inc., Tensorflow 1.8.0, and Keras 2.2.0.
