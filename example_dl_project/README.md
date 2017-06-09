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

def init_simple_cnn(config):
    pass

```

* src/training_loop.py

Resumable training loops (sometimes shipped with framework, e.g. Blocks). For instance in keras follow convention:

```{python}

def cifar_training_loop(model, train, valid, pickle_path):
    pass

```

* src/scripts

Runners (usually use vegab/argh or other command line wrapper), following convention:

```{bash}

./src/scripts/train_simple_CNN.py root results/simple_cnn/my_save_location --n_layers=10

```

* configs

Stores configs use in the project as jsons (alternative is to use config_registry).

* etc

All misc. files relevant to the project (includes meeting notes, paper sources, etc.).

## FAQ

### Why no requirements.txt or environment.yml

Not everyone uses them and it adds unnecessary friction

### Why keras specific?

It is not really keras specific. Same project structure works for other frameworks. For instance in Blocks one
wouldn't need `src/training_loop.py`.