# MLMI 4 Group Project: Bayes By Backprop

This project is an implementation of the paper
[Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424) by Blundell et al., and some further investigations based on their results.

The code is written in ``Python 3.7`` using ``TensorFlow 1.12``.

## Dates for the project

| Date       | Time            | Event                              |
| :--------: | :-------------: | ---------------------------------- |
| 25 Feb     | 14:00 - 15:30   | Poster Training Session            |
| 11 March   | 12:00           | Poster Submission Deadline         |
| 15 March   | 10:00 - 12:00   | Poster Session in JDB Seminar Room |
| 5 April    | 12:00           | Project Report Submission          |

## TODO List

 - Get everything working

## Setting Up

Clone this repo to a convenient location:

```
> git clone https://github.com/gf38/MLMI4-AML.git
```

Go to the ``project`` folder, set up a virtual environment there and activate it:

```
> cd MLMI4-AML/project
> virtualenv -p python3 env
> source venv/bin/activate
```

__Note:__ Please ensure that the virtual environment you're setting up is indeed in ``Python 3.7``
__Note:__ Here we used the name ``venv`` for the environment folder. If you choose a different name, please do not forget to add it to the ``.gitignore`` file!

Install the requirements:

```
(venv)> pip install -r requirements.txt
```

If TensorFlow fails to install, you can install it by using the command

for Linux:
```
(venv)> pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl
```

for Mac:
```
(venv)> pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl
```

__Note__: For a complete install guide, see [here](https://www.tensorflow.org/install/pip).

## Running the code

You can run the code by calling ``main.py``:

```
(venv)> python code/main.py --model=baseline
```

Currently there are two possible options for models:

| Argument        | Model          | Description                                                                 |
|-----------------|----------------|-----------------------------------------------------------------------------|
| ``baseline``    | Baseline Model | 2 Layer (800 unit) Fully Connected NN with ReLU activation MNIST classifier |
| ``bayes_mnist`` | Bayesian MNIST | work in progress                                                            |

## Running TensorBoard
To start TensorBoard first make sure to activate virtual environment and then launch TensorBoard by specifying the log directory e.g. /tmp/mnist_baseline_model/.

```
source venv/bin/activate
(venv)> tensorboard --logdir /tmp/mnist_baseline_model/
```
Access TensorBoard on the generated URL.

## Pruning Weights
Prune weights for MNIST classification by adding the flag --prune_weights. Specify the pruning_percentile in the config dictionary (classification.py).

```
(venv)> python code/classification.py --model=bayes_mnist --model_dir=models/bayes_mnist_400 --prune_weights
```

Weights from the specified directory are pruned using the given percentile. The pruned weights are stored in a directory with the same name and a suffix pruned_98 and passed to evaluation. Add the flag --no_training if you wish to prune an already trained model.



## Poster stuff
RESULTS: 
1. Reproducing the MNIST Table

*a) use the same number of data -------> 50000 samples training set, 10000 samples validation set, 10000 samples test set. 
*b) Plot: accuracy vs epochs for SFC, Dropout, Bayes
*c) Plot: Compare weight distributions associated with the three models
*d) Weight Pruning -------> plot and table



2.Regression
*a) Plot: SFC vs BBB -----> highligh the uncertainty outside the data region
*b) Heteroskedastic thing
*c) GP vs BBB
