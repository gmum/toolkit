# Example project

Simple exemplary PyTorch project template. The key design principles are:
 
 * Trainers are command line tools with a consistent interface
 * Consistent way of saving results, logs, etc
 * Generic high-quality training loop
    - Resumable
    - Results can be visualized using TensorBoard

## Command line interface

One of the key ideas is having a consistent interface to scripts; You can run all scripts as:

```
 python bin/cifar_train.py cifar10 test_run --model.n_filters=10 
```

, which slighty modifies the base config ``cifar10`` by changing n_filters to 10. After running you can find following goodies inside
the results directory ``test_run``:

```
-rw-r--r--@  1 kudkudak  staff      144 Jun 10 18:04 history.csv
-rw-r--r--@  1 kudkudak  staff       35 Jun 10 18:04 loop_state.pkl
-rw-r--r--@  1 kudkudak  staff      200 Jun 10 18:04 meta.json
-rw-r--r--@  1 kudkudak  staff  2593544 Jun 10 18:04 model.h5
-rw-r--r--@  1 kudkudak  staff     1578 Jun 10 18:04 stderr.txt
-rw-r--r--@  1 kudkudak  staff     2507 Jun 10 18:04 stdout.txt
-rw-r--r--@  1 kudkudak  staff      300 Jun 10 18:04 config.json
-rw-r--r--@  1 kudkudak  staff     1211 Jun 10 18:04 cifar_train.py
```

Here are some examples of running the trainer. For instance you can change model to LeNet by:

```
 python bin/cifar_train.py cifar10 test_run_lenet --model=lenet
```

You can also change LR schedule by:

```
 python bin/cifar_train.py cifar10 test_run_lrsch --lr_schedule="[[3, 0.1],[100, 0.01]]"
```

This is powered by a small utility called ``vegab``, which is implemented in ``src.vegab``.

## Project structure

* ./env.sh

Any relevant environment variables (including `THEANO_FLAGS`). Shouldn't be commited. Often machine specific! 

* src/data.py

Gives training dataset as iterator and test dataset as simple arrays. Usually coded with help of Fuel.

```{python}

def get_cifar(which, augment):
    return train, test

```

* src/models

Model definitions.

* src/training_loop.py

Resumable training loops (sometimes shipped with framework, e.g. Blocks). For instance in keras follow convention:

```{python}

def cifar_training_loop(model, train, valid, [other params]):
    pass

```

Note that training loop does not accept test set. This should be explicitely never look at during training,
very easy to use it (even implicitely), and thus overfit.

* bin

Different executable, including trainers

* configs

Stores configs use in the project as jsons or config_registry. This projects uses config_registry.-

* etc

All misc. files relevant to the project (includes meeting notes, paper sources, etc.).

## Requirements

See requirements.txt