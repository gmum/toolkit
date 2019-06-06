# -*- coding: utf-8 -*-
"""
A gorgeous, self-contained, training loop. Uses Poutyne implementation, but this can be swapped later.
"""

import logging
import os
import tqdm
import pickle
from functools import partial

from poutyne.framework import Model

import numpy as np
import pandas as pd
import torch
import gin

from src.callbacks.callbacks import ModelCheckpoint, LambdaCallback, History, DumpTensorboardSummaries
from src.utils import save_weights

logger = logging.getLogger(__name__)


def _construct_default_callbacks(model, optimizer, H, save_path, checkpoint_monitor, save_freq, custom_callbacks,
                                 use_tb, save_history_every_k_examples):
    callbacks = []
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
    logger.info("Saving loop_state.pkl")  # TODO: Debug?

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

    logger.info("Saved loop_state.pkl")  # TODO: Debug?


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


@gin.configurable
def training_loop(model, loss_function, metrics, optimizer, meta_data, config, save_path, train, valid, steps_per_epoch,
                  custom_callbacks=[], checkpoint_monitor="val_acc", use_tb=False, reload=True,
                  n_epochs=100, save_freq=1, save_history_every_k_examples=-1):
    callbacks = list(custom_callbacks)

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

    model = Model(model=model, optimizer=optimizer, loss_function=loss_function, metrics=metrics)
    if  torch.cuda.is_available():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info("Sending model to {}".format(device))
        model.to(device)

    _ = model.fit_generator(train,
                            initial_epoch=epoch_start,
                            steps_per_epoch=steps_per_epoch,
                            epochs=n_epochs - 1,  # Weird convention
                            verbose=1,
                            valid_generator=valid,
                            callbacks=callbacks)
