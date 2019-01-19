# icpram2019

This repository contains the implementation of the experiments proposed in the paper [Using stigmergy as a computational memory in the design of recurrent neural networks](http://www.iet.unipi.it/m.cimino/publications/cimino_pub68.pdf).
If you are interested on the **actual implementation** of the **Stigmergic Memory** please check out the  [**torchsm repository**](https://github.com/galatolofederico/torchsm)


## Installation

Clone this repository
```
git clone https://github.com/galatolofederico/icpram2019.git && cd icpram2019
```
Create a python virtualenv and activate it, make sure to use **python3** not higher than **python3.6**
```
virtualenv --python=/usr/bin/python3 env && source ./env/bin/activate
```
Install the requirements
```
pip install -r requirements.txt
```
You are ready to go!

## Contents

Each of the following script uses the [sacred](https://github.com/IDSIA/sacred/tree/master/sacred) framework to manage experiments configurations and results.
In order to set a configuration variable you need to use the sacred style
```
python3 mnist.py with config1=val1 config2=val2
```
For example
```
python3 mnist.py with batch_size=20 use_mongo=True
```

In each script you can set the following configuration variables

|Variable|Description|MNIST|MNIST Stroke|
|---|---|---|---|
|batch_size|Training batch size|20|20
|lr|Learning rate for Adam|0.001|0.001
|total_its|Number of training epochs|10|10
|hidden_dim|Number of neurons for the classification network|10|20
|hidden_layers|Number of hidden layers for the classification network|1|1
|stig_hidden_dim|Number of neurons for the mark/tick networks|20|20
|stig_hidden_layers|Number of hidden layers for the mark/tick networks|1|1
|stig_dim|Size of the Stigmergic Memory|15|30
|avg_window|Moving average window size for logging|100|100
|use_mongo|Use MongoDB Observer to log the experiments|False|False



### mnist_stroke.py

Python script to train and evaluate the Stigmergic Memory Architecture proposed in the paper against the [**MNIST digits stroke sequence data**](https://github.com/galatolofederico/pytorch-mnist-stroke) dataset.

You can additionally set the following configuration variables

|Variable|Architecture|Description|Default|
|---|---|---|---|
|arch|All|Architecture to use (stigmergic, lstm or recurrent)|stigmergic
|hidden_size|LSTM|Hidden neurons for LSTM|20
|num_layers|LSTM|Hidden layers for LSTM|1
|recurrent_dim|Recurrent|Number of recurrent connections|30
|recurrent_hidden|Recurrent|Number of hidden neurons for the recurrent network|50
|hidden_dim|Recurrent|Number of hidden neurons for the classification network|50

### mnist.py

Python script to train and evaluate the Stigmergic Memory Architecture proposed in the paper against the **Temporal MNIST Dataset**


## Citing

If you want to cite us please use this BibTeX

```
@article{galatolo_sm
,	author	= {Galatolo, Federico A and Cimino, Mario GCA and Vaglini, Gigliola}
,	title	= {Using stigmergy as a computational memory in the design of recurrent neural networks
,	journal	= {ICPRAM 2019}
,	year	= {2019}
}
```

## Contributing

This code is released under GNU/GPLv3 so feel free to fork it and submit your changes, every PR helps.  
If you need help using it or for any question please reach me at [federico.galatolo@ing.unipi.it](mailto:federico.galatolo@ing.unipi.it) or on Telegram  [@galatolo](https://t.me/galatolo)