# -*- coding: utf-8 -*-
"""
Simple data getters. Each returns iterator for train and dataset for test/valid
"""

from src.callbacks import ModelCheckpoint, LambdaCallback, History, DumpTensorflowSummaries
from src.utils import save_weights

import pandas as pd

from functools import partial

import torch
import os
import pickle
import logging
import numpy as np
logger = logging.getLogger(__name__)

def _save_loop_state(epoch, logs, save_path, save_callbacks):
    logger.info("Saving loop_state.pkl")  # TODO: Debug?

    loop_state = {"epochs_done": epoch, "callbacks": save_callbacks} # 0 index

    ## Small hack to pickle Callbacks in keras ##
    if len(save_callbacks):
        m, vd = save_callbacks[0].model, save_callbacks[0].validation_data
        for c in save_callbacks:
            c.model = None
            c.optimizer = None
            c.validation_data = None

    pickle.dump(loop_state, open(os.path.join(save_path, "loop_state.pkl"), "wb"))

    ## Revert hack ##
    if len(save_callbacks):
        for c in save_callbacks:
            c.model = m
            c.optimizer = m.optimizer
            c.validation_data = vd

    logger.info("Saved loop_state.pkl")  # TODO: Debug?


def _save_history_csv(epoch, logs, save_path, H):
    out = ""
    for key, value in logs.items():
        if isinstance(value, (int, float, complex, np.float32, np.float64)):
            out += "{key}={value}\t".format(key=key, value=value)
    logger.info(out)
    pd.DataFrame(H).to_csv(os.path.join(save_path, "history.csv"), index=False)


def _print_history_csv(epoch, logs, H):
    for key, value in logs.items():
        if isinstance(value, (int, float, complex, np.float32, np.float64)):
            if key not in H:
                H[key] = [value]
            else:
                H[key].append(value)

            # Epoch is 0 first, so 1 key. Etc
            assert len(H[key]) == epoch + 1, "Len {} = ".format(key) + str(len(H[key]))
        else:
            pass


def training_loop(model, train, valid, n_epochs, save_path, steps_per_epoch, save_freq=0,
        reload=False, custom_callbacks=[], checkpoint_monitor="val_acc"):

    # Configure a bit.
    # NOTE: We distinguish two history objects. One can have large tensors, other stores just simple floats etc.
    # NOTE: Loop state stores callbacks and epoch_id
    loop_state_path = os.path.join(save_path, "loop_state.pkl")
    history_csv_path = os.path.join(save_path, "history.csv")
    history_pkl_path = os.path.join(save_path, "history.pkl")
    model_last_epoch_path = os.path.join(save_path, "model_last_epoch.pt")
    callbacks = list(custom_callbacks)

    # Reloading procedure (TODO: move out)
    if reload:
        if not os.path.exists(model_last_epoch_path) or not os.path.exists(loop_state_path):
            raise IOError("Failed to find last epoch model or loop state")
        # Reload everything (model, optimizer, loop state)
        logger.warning("Reloading weights!")
        checkpoint = torch.load(model_last_epoch_path)
        model.model.load_state_dict(checkpoint['model'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])
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
        epoch_start = loop_state['epochs_done'] # 0 index
        if not len(H[next(iter(H))]) == loop_state['epochs_done']:
            raise IOError("Mismatch between saved history and epochs recorded. "
                          "Found len(H)={0} and epoch_start={1} "
                          "Run was likely interrupted incorrectly and cannot be rerun.".format(len(H[next(iter(H))]), epoch_start))

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
    else:
        # Clean-up a bit..
        logger.info("Removing " + history_pkl_path)
        logger.info("Removing " + history_csv_path)
        os.system("rm " + history_pkl_path)
        os.system("rm " + history_csv_path)

        epoch_start = 0
        H = {}

    # Add callbacks (TODO: move out)
    callbacks.append(LambdaCallback(on_epoch_end=partial(_save_history_csv, save_path=save_path, H=H)))
    callbacks.append(LambdaCallback(on_epoch_end=partial(_print_history_csv, H=H)))
    callbacks.append(History(save_path=history_pkl_path))
    callbacks.append(ModelCheckpoint(monitor=checkpoint_monitor,
        model=model,
        optimizer=model.optimizer,
        save_best_only=True,
        mode='max',
        filepath=os.path.join(save_path, "model_best_val.pt")))
    if save_freq > 0:
        def save_weights_fnc(epoch, logs):
            if epoch % save_freq == 0:
                logger.info("Saving model from epoch " + str(epoch))
                save_weights(model.model, model.optimizer, os.path.join(save_path, "model_last_epoch.pt"))

        callbacks.append(LambdaCallback(on_epoch_end=save_weights_fnc))
    # Always save from first epoch
    def save_weights_fnc(logs=None):
        logger.info("Saving model from beginning")
        save_weights(model.model, model.optimizer, os.path.join(save_path, "init_weights.pt"))

    callbacks.append(LambdaCallback(on_train_begin=save_weights_fnc))
    callbacks.append(LambdaCallback(on_epoch_end=partial(_save_loop_state, save_callbacks=custom_callbacks,
        save_path=save_path)))

    _ = model.fit_generator(train,
        initial_epoch=epoch_start,
        steps_per_epoch=steps_per_epoch,
        epochs=n_epochs,
        verbose=1,
        valid_generator=valid,
        callbacks=callbacks)
