# Vanilla PyTorch project template

Simple machine learning project template based on PyTorch. 

If you are impatient just jump to the tutorial at the end of this README.

## Introduction

The main goal of this template is to make easy following the state-of-the-art good practices for a machine learning project. This includes reducing boilerplate, or keeping config handling simple and consistent.

First, this template includes a minimal trainer ``bin/train.py`` that has:

* A gorgeous training loop (generic training script, checkpointing, callbacks, etc.)
    - We use Poutyne (only in sec.training_loop). Can be swapped for something else.
* Beautiful config handling
    - We use gin for this
* Amazing automatic saving of logs and other auxiliary files

This repo also ships with:

* An example experiment in `experiments/tune_lr`
* Environment configuration 

In the rest of this document we walk through all key design principles and how we realize them. Finally, there is a quick tutorial.

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

* Each experiment should be as self-contained as possible, e.g. include runner, plotting utilities, etc. 

    - See `tune_lr` for an example. 
    
* Test everything easily testable
   
    - We have asserts sprinkled in few places in code, but as many as we should.
    

## Tutorial: single train

Take the following steps:

1. Install a minimal conda environment: ``conda env create --file e.yml``.

2. Activate the environment: ``source e.sh``.

3. Train a small CNN on Cifar10: ``bin/train.py save_to_folder configs/cnn.gin``.

4. Run ``tensorboard --tb=save_to_folder`` to visualize the learning curves.

5. Continue training for more epochs: ``bin/train.py save_to_folder configs/cnn.gin -b="training_loop.n_epochs=5;training_loop.reload=True"``.

## Tutorial: experiment

Experiment conceptually is a self-contained binary. It can be run, paused, visualized, etc. 

This experiment will tune LR for the small CNN on Cifar10. Take the following steps:

1. 