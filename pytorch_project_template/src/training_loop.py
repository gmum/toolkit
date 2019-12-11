# -*- coding: utf-8 -*-
"""
A gorgeous and self-contained training loop.
"""

import logging
import os
import tqdm
import pickle

from functools import partial
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
import gin

from src.callbacks.callbacks import ModelCheckpoint, LambdaCallback, History, DumpTensorboardSummaries, BaseLogger
from src.utils import save_weights

logger = logging.getLogger(__name__)


def _construct_default_callbacks(model, optimizer, H, save_path, checkpoint_monitor, save_freq, custom_callbacks,
                                 use_tb, save_history_every_k_examples):
    callbacks = []
    callbacks.append(BaseLogger())
    callbacks.append(LambdaCallback(on_epoch_end=partial(_append_to_history_csv, H=H)))
    callbacks.append(LambdaCallback(on_epoch_end=partial(_save_history_csv, save_path=save_path, H=H)))
    callbacks.append(History(save_every_k_examples=save_history_every_k_examples))
    callbacks.append(ModelCheckpoint(monitor=checkpoint_monitor,
                                     save_best_only=True,
                                     mode='max',
                                     filepath=os.path.join(save_path, "model_best_val.pt")))
    if save_freq > 0:
        def save_weights_fnc(epoch, logs):
            if epoch % save_freq == 0:
                logger.info("Saving model from epoch " + str(epoch))
                save_weights(model, optimizer, os.path.join(save_path, "model_last_epoch.pt"))

        callbacks.append(LambdaCallback(on_epoch_end=save_weights_fnc))

    if use_tb:
        callbacks.append(DumpTensorboardSummaries())
    callbacks.append(LambdaCallback(on_epoch_end=partial(_save_loop_state, save_callbacks=custom_callbacks,
                                                         save_path=save_path)))
    return callbacks


def _save_loop_state(epoch, logs, save_path, save_callbacks):
    loop_state = {"epochs_done": epoch, "callbacks": save_callbacks}  # 0 index

    ## A small hack to pickle callbacks ##
    if len(save_callbacks):
        m, opt, md = save_callbacks[0].get_model(), save_callbacks[0].get_optimizer(), save_callbacks[0].get_meta_data()
        for c in save_callbacks:
            c.set_model(None, ignore=False)  # TODO: Remove
            c.set_optimizer(None)
            c.set_params(None)  # TODO: Remove
            c.set_meta_data(None)
    pickle.dump(loop_state, open(os.path.join(save_path, "loop_state.pkl"), "wb"))
    if len(save_callbacks):
        for c in save_callbacks:
            c.set_model(m)
            c.set_optimizer(opt)
            c.set_meta_data(md)


def _save_history_csv(epoch, logs, save_path, H):
    out = ""
    for key, value in logs.items():
        if isinstance(value, (int, float, complex, np.float32, np.float64)):
            out += "{key}={value}\t".format(key=key, value=value)
    logger.info(out)
    logger.info("Saving history to " + os.path.join(save_path, "history.csv"))
    pd.DataFrame(H).to_csv(os.path.join(save_path, "history.csv"), index=False)


def _append_to_history_csv(epoch, logs, H):
    for key, value in logs.items():
        if isinstance(value, (int, float, complex, np.float32, np.float64)):
            if key not in H:
                H[key] = [value]
            else:
                H[key].append(value)

            # Epoch is 0 first, so 1 key. Etc
            assert len(H[key]) == epoch + 1, "Len H[{}] is {}, expected {} ".format(key, len(H[key]), epoch + 1)
        else:
            pass


def _reload(model, optimizer, save_path, callbacks):
    model_last_epoch_path = os.path.join(save_path, "model_last_epoch.pt")
    loop_state_path = os.path.join(save_path, "loop_state.pkl")
    history_csv_path = os.path.join(save_path, "history.csv")

    if not os.path.exists(model_last_epoch_path) or not os.path.exists(loop_state_path):
        logger.warning("Failed to find last epoch model or loop state")
        return {}, 0

    # Reload everything (model, optimizer, loop state)
    logger.warning("Reloading weights!")
    checkpoint = torch.load(model_last_epoch_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
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

    return H, epoch_start

@contextmanager
def _set_training_mode(model, training):
    old_training = model.training
    model.train(training)
    with torch.set_grad_enabled(training):
        yield
    model.train(old_training)


def _training_loop(model, valid_generator, train_generator, optimizer, loss_function, initial_epoch, epochs, callbacks,
                   metrics=[], device="cuda"):
    """
    Internal implementation of the training loop.

    Notes
    -----
    Loosly based on https://github.com/keras-team/keras/keras/engine/training_generator.py
    """
    model.to(device)

    for c in callbacks:
        c.on_train_begin(model)

    for epoch in range(initial_epoch, epochs):
        epoch_logs = {}

        for c in callbacks:
            c.on_epoch_begin(epoch, epoch_logs)

        # Train an epoch
        with _set_training_mode(model, True):
            for batch_id, (x_train, y_train) in enumerate(train_generator):
                x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
                x_train, y_train = x_train.to(device), y_train.to(device)

                batch_logs = {"size": len(x_train), "batch": batch_id}

                for c in callbacks:
                    c.on_batch_begin(batch=batch_id, logs=batch_logs)

                optimizer.zero_grad()

                outputs = model(x_train)
                loss = loss_function(outputs, y_train)
                loss.backward()
                optimizer.step()

                # Update logs
                for m in metrics:
                    batch_logs[m.__name__] = float(m(outputs, y_train))
                batch_logs['loss'] = loss.item()

                for c in callbacks:
                    c.on_batch_end(batch=batch_id, logs=batch_logs)

        # Validate
        with _set_training_mode(model, False):
            val = defaultdict(float)
            seen = 0
            for x_valid, y_valid in valid_generator:
                x_valid, y_valid = torch.from_numpy(x_valid), torch.from_numpy(y_valid)
                x_valid, y_valid = x_valid.to(device), y_valid.to(device)
                seen += len(x_valid)
                outputs = model(x_valid)
                val['loss'] += loss_function(outputs, y_valid) * len(x_valid)
                for m in metrics:
                    val[m.__name__] += float(m(outputs, y_valid)) * len(x_valid)
            for k in val:
                epoch_logs['val_' + k] = val[k] / seen

        for c in callbacks:
            c.on_epoch_end(epoch, epoch_logs)

        logger.info('End of epoch {}, loss={}, val_loss={}'.format(epoch, epoch_logs['loss'], epoch_logs['val_loss']))

    for c in callbacks:
        c.on_train_end(model)


@gin.configurable
def training_loop(model, loss_function, metrics, optimizer, meta_data, config, save_path, train, valid,
                  custom_callbacks=[], checkpoint_monitor="val_acc", use_tb=False, reload=True,
                  n_epochs=100, save_freq=1, save_history_every_k_examples=-1, device=None):
    callbacks = list(custom_callbacks)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if reload:
        H, epoch_start = _reload(model, optimizer, save_path, callbacks)
    else:
        save_weights(model, optimizer, os.path.join(save_path, "init_weights.pt"))

        history_csv_path, history_pkl_path = os.path.join(save_path, "history.csv"), os.path.join(save_path,
                                                                                                  "history.pkl")
        logger.info("Removing {} and {}".format(history_pkl_path, history_csv_path))
        os.system("rm " + history_pkl_path)
        os.system("rm " + history_csv_path)
        H, epoch_start = {}, 0

    callbacks += _construct_default_callbacks(model, optimizer, H, save_path, checkpoint_monitor,
                                              save_freq, custom_callbacks, use_tb,
                                              save_history_every_k_examples)

    # Configure callbacks
    for clbk in callbacks:
        clbk.set_save_path(save_path)
        clbk.set_model(model, ignore=False)  # TODO: Remove this trick
        clbk.set_optimizer(optimizer)
        clbk.set_meta_data(meta_data)
        clbk.set_config(config)

    _training_loop(model, valid, train, optimizer, loss_function, epoch_start, n_epochs, callbacks,
                   metrics=metrics, device=device)