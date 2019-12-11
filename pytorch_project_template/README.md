# PyTorch project template

Simple machine learning project template based on PyTorch. 

If you are impatient just jump to the tutorial at the end of this README.

A small warning is due. This has been rewritten recently to remove the dependency on PyToune. It might include some bugs.

## Introduction

The main goal of this template is to make easy following the state-of-the-art good practices for a machine learning project. This includes reducing boilerplate, or keeping config handling simple and consistent.

First, this template includes a minimal trainer ``bin/train.py`` that has:

* A gorgeous training loop (generic training script, checkpointing, callbacks, etc.)
* Beautiful config handling
    - We use gin for this
* Amazing automatic saving of logs and other auxiliary files

This repo also ships with:

* An example experiment in `experiments/tune_lr`
* Environment configuration 

In the rest of this document we walk through all the key design principles and how we implement them. Finally, there is a quick tutorial.

## Good practices 

Here is a non-exhaustive list of good practices that this project tries to implement. They are based on
state-of-the-art ideas in the community about how to organise a machine learning project. 

Do you have other ideas? Please open an issue and let's discuss. Here are ours:

* Use a consistent environment 

    - We use conda. See `e.sh`. You should source it each time you start working on the repo. You shouldn't commi it. 

* Use a common training loops for all trainings. Use callbacks. Everything should be resumable.

    - See `src/training_loop.py`

* Structure code so that you can plug-and-play different models, datasets, callbacks, etc. 

    - See `data`, `models` and `callbacks` modules. See how `bin/train.py` allows for easy gin-based configuration.

* Separate functionalities as binaries. Motivation is similar to tooling like `ls` in unix systems.

    - See `bin` folder

* Store common configs. 
    
    - See `configs` folder.

* Each run should have a dedicated folder with everything in one place, always in the same format (including logs, used command). 

    - See `results/example_run` folder for example files that are produced.

* Each experiment should be as self-contained as possible, e.g. include runner, plotting utilities, a README file, etc. 

    - See `experiments/tune_lr` for an example. 
    
* Test everything easily testable
   
    - We have asserts sprinkled across the code, but probably not as many as we should.
    

## Tutorial: single training

Take the following steps:

1. Install a minimal conda environment: ``conda env create --file e.yml``.

2. Activate the environment: ``source e.sh``.

3. Train on few batches a CNN on Cifar10: ``bin/train.py save_to_folder configs/cnn.gin``.

4. Run ``tensorboard --logdir=save_to_folder`` to visualize the learning curves.

Configuration is done using gin. This allows for a flexible configuration of training. For instance, to continue training for more epochs you can run: ``bin/train.py save_to_folder configs/cnn.gin -b="training_loop.n_epochs=5#training_loop.reload=True"``.

Note: training won't reach sensible accuracies. This is on purpose so that the demonstration works on small machines. For a bit more realistic training configuration see `configs/cnn_full.gin`.

## Tutorial: experiment example

Experiment conceptually is a list of shell jobs. For convenience this can be wrapped using a python script that prepares jobs, analyses the runs, stores configs, etc. 

We ship an example experiment, where we tune LR for the small CNN on Cifar10. Here is the typical workflow:

1. Prepare experiments: `python experiments/tune_lr/main.py prepare`

2. See prepare configs: `ls experiments/tune_lr/large/configs`

3. Run experiments: `bash experiments/tune_lr/large/batch.sh`

4. See runs: `ls $RESULTS_DIR/tune_lr/large`

    5. Process experiment results: `python experiments/tune_lr/main.py report`. Bonus for OSX users: To enable plotting in iterm install ``pip install itermplot``, and uncomment the appropriate line in ``e.sh```.

6. Take a look at the main.py source code to understand better the logic.

Note that running a list of shell jobs can be done using a scheduler. This is best if you develop your own
solution for runnning efficiently such a list.

