
# Getting started

You can get the code for different parts of this project using the following command.

```console
git clone https://github.com/algo-cancer/PhyloM.git
```
The list of packages and their version installed in our system for running these codes is provided in **packages.txt** file.
For data construction, the ms program is required from https://uchicago.app.box.com/s/l3e5uf13tikfjm7e1il1eujitlsjdx13. This program should be compiled based on its own instruction.

# Branching Inference
The code for this part is placed under Branching_Inference folder.

See https://github.com/algo-cancer/PhyloM/blob/master/Branching_Inference/README.md for the instructions.

# Noise Inference
The code for this part has been provided under Noise_Inference folder.
See https://github.com/algo-cancer/PhyloM/blob/master/Noise_Inference/README.md for the instructions.

# Noise Elimination
The code for this part could be found under Noise_Elimination folder.
See https://github.com/algo-cancer/PhyloM/blob/master/Noise_Elimination/README.md for the instructions.



# System configs
All the experiments in this work are performed using **Carbonate**, a computer cluster at Indiana University. We used deep learning (DL) nodes in Carbonate.
These nodes features 12 GPU-accelerated Lenovo ThinkSystem SD530 , each equipped with two Intel Xeon Gold 6126 12-core CPUs, two NVIDIA GPU accelerators (eight with Tesla P100s; four with Tesla V100s), and 192 GB of RAM. A Red Hat Enterprise 7.x is running on these nodes and we used **Python 3.7.3** :: Anaconda, Inc., **Tensorflow 1.13.1**, and **Keras 2.2.4**.

# Contact
Please email us at <esadeqia@iu.edu> or <mhaghir@iu.edu> for any questions.
