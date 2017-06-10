# Example project

Simple project training CNN on CIFAR10 illustrating well structured DL project. 

## Project structure

* ./env.sh

Any relevant environment variables (including `THEANO_FLAGS`). Shouldn't be commited. 

* src/data.py

Gives training dataset as iterator and test dataset as simple arrays. Usually coded with help of Fuel.

```{python}

def get_cifar(which, augment):
    return train, test

```

* src/models.py

Model definitions.

```{python}

def build_simple_cnn(config):
    pass

```

Note: if your model is complicated (for instance has custom inference procedure), it is a good
idea to wrap model in a class supporting these methods. For instance you can construct new block in Blocks
or new Model in keras.

* src/training_loop.py

Resumable training loops (sometimes shipped with framework, e.g. Blocks). For instance in keras follow convention:

```{python}

def cifar_training_loop(model, train, valid, [other params]):
    pass

```

Note that training loop does not accept test set. This should be explicitely never look at during training,
very easy to use it (even implicitely), and thus overfit.

* src/scripts

Runners (usually use vegab/argh or other command line wrapper), following convention:

```{bash}

./src/scripts/train_simple_CNN.py root results/simple_cnn/my_save_location --n_layers=10

```

Any DL code should be resumable by default.

* configs

Stores configs use in the project as jsons or config_registry. This projects uses config_registry.-

* etc

All misc. files relevant to the project (includes meeting notes, paper sources, etc.).

## FAQ

### Why no requirements.txt or environment.yml

Not everyone uses them and it adds unnecessary friction

### Why keras specific?

It is not really keras specific. Same project structure works for other frameworks. For instance in Blocks one
wouldn't need `src/training_loop.py`.