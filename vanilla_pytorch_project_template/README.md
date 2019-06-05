# Vanilla PyTorch project template

Simple machine learning project template based on PyTorch. 

## Single train dive-in

Take the following steps:

1. Install a minimal conda environment: ``conda env create --file e.yml``.

2. Activate the environment: ``source e.sh``.

3. Train a small CNN on Cifar10: ``bin/train.py save_to_folder configs/cnn.gin``.

4. Run ``tensorboard --tb=save_to_folder`` to visualize the learning curves.

5. Continue training for more epochs: ``bin/train.py save_to_folder configs/cnn.gin -b="training_loop.n_epochs=5;training_loop.reload=True"``.

## Experiment dive-in

Experiment conceptually is a self-contained binary. It can be run, paused, visualized, etc. 

This experiment will tune LR for the small CNN on Cifar10. Take the following steps:

1. 


## Introduction

The main goal of this template is to make easy following good practices for setting-up a machine learning project. The main design objectives are to reduce possibility of bugs, and speed up the development process.

First, this template includes a minimal trainer ``bin/train.py`` that has:

* Training loop (generic training script, checkpointing, callbacks, etc.)
* Config handling using Google's gin 
* Saving logs and other auxiliary files automagically

We also include:

* An example experiment in `experiments/tune_lr`
* Environment configuration 

In the rest of this document we describe the project structure, and then walk through all key
design principles.

## Good practices 

Here is a non-exhaustive list of good practices that this project structure is model after. Then we describe how this project encourages it. These
are motivated from the perspective of making few bugs. 

This list should be expanded as we learn more about what is a good repository structure.

* Use a consistent environment 

    - See `e.sh`. You should source it each time you start working on the repo. You shouldn't commi it. 

* Use a common training loops between scripts. Everything should be resumable.

    - See `src/training_loop.py`

* Structure code so that you can plug-and-play different models, datasets, callbacks, etc. 

    - See `data`, `models` and `callbacks` modules. See how `bin/train.py` allows for easy gin-based configuration.

* Separate functionalities as binaries. This is similar to tooling like `ls` in unix systems.

    - See `bin` folder

* Store common configs. 
    
    - See `configs` folder.

* Each run = one folder with everything (including logs, used command). 

    - See `results/example_run` folder for example files that are produced.

* Each experiment should be as self-contained as possible, e.g. include runner, plotting utilities, etc. 

    - See `tune_lr` for an example. 