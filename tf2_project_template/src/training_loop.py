# -*- coding: utf-8 -*-
"""
Self-contained training loop for the project.
"""

import logging
import os
import pickle
from collections import defaultdict
from functools import partial
import gin
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.metrics import Metric as keras_metric

from src.callbacks import ModelCheckpoint, LambdaCallback, History, DumpTensorboardSummaries, BaseLogger, Heartbeat, \
    MetaSaver, MeasureTime
from src.utils import save_model, restore_model_and_optimizer, restore_model, configure_neptune_exp

logger = logging.getLogger(__name__)


def _construct_default_callbacks(model, optimizer, H, save_path, checkpoint_monitor, save_freq, custom_callbacks,
                                 use_tb, save_history_every_k_examples):
    callbacks = []
    callbacks.append(MetaSaver())
    callbacks.append(BaseLogger())
    callbacks.append(MeasureTime())
    callbacks.append(Heartbeat())
    callbacks.append(LambdaCallback(on_epoch_end=partial(_append_to_history_csv, H=H)))
    callbacks.append(LambdaCallback(on_epoch_end=partial(_save_history_csv, save_path=save_path, H=H)))
    callbacks.append(History(save_every_k_examples=save_history_every_k_examples))

    if len(checkpoint_monitor):
        callbacks.append(ModelCheckpoint(monitor=checkpoint_monitor,
                                         save_best_only=True,
                                         mode='max',
                                         filepath=os.path.join(save_path, "model_best_val")))
    if save_freq > 0:
        def save_weights_fnc(epoch, logs):
            if epoch % save_freq == 0:
                logger.info("Saving model from epoch " + str(epoch))
                save_model(model, optimizer, os.path.join(save_path, "model_last_epoch"))

        callbacks.append(LambdaCallback(on_epoch_end=save_weights_fnc))

    if use_tb:
        callbacks.append(DumpTensorboardSummaries())
    callbacks.append(LambdaCallback(on_epoch_end=partial(_save_loop_state, save_callbacks=custom_callbacks,
                                                         save_path=save_path)))
    return callbacks


def _save_loop_state(epoch, logs, save_path, save_callbacks):
    loop_state = {"epochs_done": epoch, "callbacks": save_callbacks}
    pickle.dump(loop_state, open(os.path.join(save_path, "loop_state.pkl"), "wb"))


def _save_history_csv(epoch, logs, save_path, H):
    out = ""
    keys_to_save = []
    for key, value in logs.items():
        if isinstance(value, (int, float, complex, np.float32, np.float64)):
            out += "{key}={value}\t".format(key=key, value=value)
            keys_to_save.append(key)
    logger.info(out)
    logger.info("Saving history to " + os.path.join(save_path, "history.csv"))
    try:
        pd.DataFrame(H)[keys_to_save].to_csv(os.path.join(save_path, "history.csv"), index=False)
    except ValueError:
        logger.warning("Couldnt' save history.csv. Please read in history.pkl")


def _append_to_history_csv(epoch, logs, H):
    for key, value in logs.items():
        if isinstance(value, (int, float, complex, np.float32, np.float64)):
            if key not in H:
                H[key] = [value]
            else:
                H[key].append(value)

            assert len(H[key]) == epoch + 1, "Len H[{}] is {}, expected {} ".format(key, len(H[key]), epoch + 1)
        else:
            pass


def _reload(model, optimizer, save_path, callbacks):
    model_last_epoch_path = os.path.join(save_path, "model_last_epoch")
    loop_state_path = os.path.join(save_path, "loop_state.pkl")
    history_csv_path = os.path.join(save_path, "history.csv")

    if not os.path.exists(model_last_epoch_path) or not os.path.exists(loop_state_path):
        logger.warning("Failed to find last epoch model or loop state")
        return {}, 0

    # Reload everything (model, optimizer, loop state)
    logger.warning("Reloading weights!")
    model, optimizer = restore_model_and_optimizer(model, optimizer, model_last_epoch_path)
    logger.info("Reloading loop state!")
    loop_state = pickle.load(open(loop_state_path, 'rb'))
    logger.info("Reloading history!")
    H = pd.read_csv(history_csv_path)
    H = {col: list(H[col].values) for col in H.columns}
    print(H)
    logger.info("Done reloading!")

    # Small back-up
    os.system("cp " + os.path.join(save_path, "history.pkl") + " " + os.path.join(save_path, "history.pkl.bckp"))

    # Setup the rest
    epoch_start = loop_state['epochs_done'] + 1
    if not len(H[next(iter(H))]) == loop_state['epochs_done'] + 1:
        raise IOError("Mismatch between saved history and epochs recorded. "
                      "Found len(H)={0} and epoch_start={1} "
                      "Run was likely interrupted incorrectly and cannot be rerun.".format(len(H[next(iter(H))]),
                                                                                           epoch_start))

    # Load all callbacks from the loop_state
    for e, e_loaded in zip(callbacks, loop_state['callbacks']):
        assert type(e) == type(e_loaded)
        if hasattr(e, "__setstate__"):
            e.__setstate__(e_loaded.__dict__)
        else:
            e.__dict__.update(e_loaded.__dict__)

    # Some diagnostics
    logger.info(loop_state)
    for k in H:
        logger.info((k, len(H)))
        break

    logger.info("epoch_start={}".format(epoch_start))

    return model, optimizer, H, epoch_start


def _to_infinite_iterator(it):
    while True:
        for bundle in it:
            yield bundle


def evaluate(model, data_generators, loss_function, metrics):
    results_all = defaultdict(float)
    for dataset_id in range(len(data_generators)):
        results = defaultdict(float)
        seen = 0

        for m in metrics:
            if isinstance(m, keras_metric):
                m.reset_states()

        for x, y in data_generators[dataset_id]:
            outputs = model.predict(x)

            if isinstance(x, dict):
                for k in x:
                    seen += len(x[k])
                    # print(("Seen", len(x[k])))
                    assert len(outputs) == len(x[k])
                    break
            else:
                seen += len(x)
                assert len(outputs) == len(x)

            results[f'loss:{dataset_id}'] += float(np.sum(loss_function(y, outputs)))

            for m in metrics:
                results[m.__name__ + ":" + str(dataset_id)] += float(np.sum(m(y, outputs)))

        for m in metrics:
            results[m.__name__ + ":" + str(dataset_id)] /= seen
        results_all.update(results)

    # Just to make sure
    for m in metrics:
        if isinstance(m, keras_metric):
            m.reset_states()

    return results_all


def _train_on_batch(model, optimizer, x_trains, y_trains, metrics, loss_function,
                    weight_decay=0):
    batch_logs = {}

    with tf.GradientTape(persistent=True) as tape:  # TODO: Fix persistency
        loss = 0
        all_outputs = []
        all_inputs = []
        for dataset_id, (x_train, y_train) in enumerate(zip(x_trains, y_trains)):
            outputs = model(x_train, training=True)

            all_outputs.append(outputs)
            all_inputs.append(x_train)

            loss += tf.math.reduce_mean(loss_function(y_train, outputs))

            # Update logs
            for m in metrics:
                if isinstance(m, str):
                    continue
                batch_logs[m.__name__ + ":" + str(dataset_id)] = tf.math.reduce_mean(
                    m(outputs, y_train))  # If not scalar, reduce it

            batch_logs[f'loss:{dataset_id}'] = loss


    grads = tape.gradient(loss, model.trainable_weights)

    if weight_decay > 0:
        grads = [g + weight_decay * p for (g, p) in zip(grads, model.trainable_weights)]

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return batch_logs


@tf.function
def _train_on_batch_optimized(model, optimizer, x_trains, y_trains, metrics,
                              loss_function, weight_decay=0):
    return _train_on_batch(model=model,
                           optimizer=optimizer,
                           x_trains=x_trains, y_trains=y_trains,
                           metrics=metrics,
                           loss_function=loss_function, weight_decay=weight_decay)


def _training_loop(model, datasets, optimizer, loss_function, initial_epoch, epochs, callbacks,
                   steps_per_epoch, train_on_batch, evaluate_model, metrics=[], weight_decay=0,
                   evaluation_freq=1):
    """
    Internal implementation of the training loop.
    """
    tf.keras.backend.set_learning_phase(1)

    train_generators = [_to_infinite_iterator(d[0]) for d in datasets]
    valid_generators = [d[1] for d in datasets]

    for c in callbacks:
        c.on_train_begin(model)

    cumulative_batch_id = 0

    for epoch in range(initial_epoch, epochs):
        logger.info(f"Start epoch {epoch}")

        epoch_logs = {}

        for c in callbacks:
            c.on_epoch_begin(epoch, epoch_logs)
        for batch_id in range(steps_per_epoch):
            cumulative_batch_id += 1
            batch_logs = {}

            x_trains, y_trains, y_trains_a, y_trains_b, lams = [], [], [], [], []

            for dataset_id in range(len(datasets)):
                x_train, y_train = next(train_generators[dataset_id])

                x_trains.append(x_train)
                y_trains.append(y_train)

                if isinstance(x_train, dict):
                    batch_logs.update({"size:" + str(dataset_id): len(list(x_train.values())[0])})
                else:
                    batch_logs.update({"size:" + str(dataset_id): len(x_train)})

            for c in callbacks:
                c.on_batch_begin(batch=batch_id, logs=batch_logs)

            batch_logs_step = train_on_batch(model, optimizer, x_trains, y_trains,
                                             metrics,
                                             loss_function,
                                             weight_decay=weight_decay)

            batch_logs.update(batch_logs_step)
            for k in batch_logs:
                if hasattr(batch_logs[k], "numpy"):
                    batch_logs[k] = batch_logs[k].numpy()

                # if isinstance(batch_logs[k], tf.Tensor):
                if (hasattr(batch_logs[k], 'ndim') and batch_logs[k].ndim > 0) or isinstance(batch_logs[k], list):
                    batch_logs[k] = batch_logs[k]  # .numpy()
                    if isinstance(batch_logs[k], list):
                        batch_logs[k] = np.array(batch_logs[k])
                else:
                    batch_logs[k] = float(batch_logs[k])

            for c in callbacks:
                c.on_batch_end(batch=batch_id, logs=batch_logs)

        if evaluation_freq > 0 and (epoch % evaluation_freq == 0 or epoch == epochs - 1):
            tf.keras.backend.set_learning_phase(0)
            val_results = evaluate_model(model, valid_generators, loss_function, metrics)
            tf.keras.backend.set_learning_phase(1)
            for k, v in val_results.items():
                epoch_logs[f'val_{k}'] = v
        else:
            if evaluation_freq > 0:
                for k in previous_epoch_logs:
                    if k not in epoch_logs:
                        epoch_logs[k] = np.nan

        for c in callbacks:
            c.on_epoch_end(epoch, epoch_logs)

        logger.info('End of epoch {}, loss={}'.format(epoch, epoch_logs['loss:0']))

        previous_epoch_logs = dict(epoch_logs)

    for c in callbacks:
        c.on_train_end(model)


@gin.configurable
def training_loop(model, loss_function, metrics, optimizer,
                  config, save_path, datasets, steps_per_epoch, seed,
                  custom_callbacks=[], checkpoint_monitor="val_categorical_accuracy:0",
                  use_tb=False, reload=False, evaluation_freq=1,
                  n_epochs=100, save_freq=1, save_history_every_k_examples=1,
                  load_weights_from="", load_weights_and_optimizer_from="",
                  weight_decay=0, load_classifier=True):
    if load_weights_and_optimizer_from != "":
        assert load_weights_from == ""
        assert load_weights_and_optimizer_from.endswith("h5")
        logger.info(f"load_weights_and_optimizer_from={load_weights_and_optimizer_from}")
        _, optimizer = restore_model_and_optimizer(model, optimizer, load_weights_and_optimizer_from)
        logger.info("Loaded optimizer")
        model.load_weights(load_weights_and_optimizer_from, by_name=True)
        model.optimizer = optimizer
    elif load_weights_from != "":
        assert load_weights_and_optimizer_from == ""
        model.load_weights(load_weights_from)

    if reload:
        assert load_classifier is True
        logger.warning("Only custom_callbacks can be reloaded for now")
        previous_model = model
        model, optimizer, H, epoch_start = _reload(model, optimizer, save_path, custom_callbacks)
        del previous_model
        logger.warning("Changed model reference internally in the training loop!")
    else:
        if hasattr(model, "compile"):
            model.compile(optimizer=optimizer,
                          loss=loss_function,
                          metrics=[m for m in metrics if not isinstance(m, str)])  # FIXME

        save_model(model, optimizer, os.path.join(save_path, "init_weights"))

        history_csv_path, history_pkl_path = os.path.join(save_path, "history.csv"), os.path.join(save_path,
                                                                                                  "history.pkl")
        logger.info("Removing {} and {}".format(history_pkl_path, history_csv_path))
        os.system("rm " + history_pkl_path)
        os.system("rm " + history_csv_path)
        H, epoch_start = {}, 0


    callbacks = list(custom_callbacks)
    callbacks += _construct_default_callbacks(model, optimizer, H, save_path, checkpoint_monitor,
                                              save_freq, custom_callbacks, use_tb,
                                              save_history_every_k_examples)

    # Configure callbacks
    for clbk in callbacks:
        clbk.set_save_path(save_path)
        clbk.set_optimizer(optimizer)
        clbk.set_model(model)
        clbk.set_seed(seed)
        clbk.set_datasets(datasets)
        clbk.set_config(config)
        clbk.set_callbacks(callbacks)

    _training_loop(model, datasets, optimizer, loss_function, epoch_start, n_epochs, callbacks,
                   steps_per_epoch, train_on_batch=_train_on_batch_optimized,
                   evaluate_model=evaluate,
                   metrics=metrics, evaluation_freq=evaluation_freq,
                   weight_decay=weight_decay)

    if save_freq != -1:
        save_model(model, optimizer, os.path.join(save_path, "model_last_epoch"))
