# -*- coding: utf-8 -*-
"""
Simple data getters. Each returns iterator for train and dataset for test/valid
"""

from src.callbacks import ModelCheckpoint,LambdaCallback

import pandas as pd

from functools import partial

import os
import pickle
import logging
logger = logging.getLogger(__name__)

from src.callbacks import LambdaCallback, DumpTensorflowSummaries

def _save_loop_state(epoch, logs, save_path, save_callbacks):
    logger.info("Saving loop_state.pkl")  # TODO: Debug?

    loop_state = {"last_epoch_done_id": epoch, "callbacks": save_callbacks}

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


def _save_and_print_history(epoch, logs, save_path):
    history_path = os.path.join(save_path, "history.csv")
    if os.path.exists(history_path):
        H = pd.read_csv(history_path)
        H = {col: list(H[col].values) for col in H.columns}
    else:
        H = {}

    # Report
    out = ""
    for key, value in logs.items():
        out += "{key}={value}\t".format(key=key, value=value)
    logger.info(out)

    # Save
    for key, value in logs.items():
        if key not in H:
            H[key] = [value]
        else:
            H[key].append(value)

    pd.DataFrame(H).to_csv(os.path.join(save_path, "history.csv"), index=False)

def training_loop(model, train, valid, n_epochs, save_path, steps_per_epoch,
        reload=False, custom_callbacks=[], checkpoint_monitor="val_acc"):


    loop_state = {'last_epoch_done_id': -1}
    loop_state_path = os.path.join(save_path, "loop_state.pkl")
    model_last_epoch_path = os.path.join(save_path, "model_last_epoch.h5")
    callbacks = list(custom_callbacks)

    # if reload and os.path.exists(model_last_epoch_path) and os.path.exists(loop_state_path):
    #
    #     # TODO(kudkudak): Add checking
    #
    #     logger.warning("Reloading weights!")
    #     model.load_weights(model_last_epoch_path)
    #
    #     # code from https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/utils/save_load_utils.py
    #     with h5py.File(model_last_epoch_path) as f:
    #         if 'optimizer_weights' in f:
    #             # build train function (to get weight updates)
    #             model._make_train_function() # Note: might need call to model
    #             optimizer_weights_group = f['optimizer_weights']
    #             optimizer_weight_names = [n.decode('utf8') for n in optimizer_weights_group.attrs['weight_names']]
    #             logger.info(optimizer_weight_names)
    #             optimizer_weight_values = [optimizer_weights_group[n] for n in optimizer_weight_names]
    #             model.optimizer.set_weights(optimizer_weight_values)
    #         else:
    #             logger.warning("No optimizer weights in wieghts file!")
    #
    #     logger.info("Reloading loop state!")
    #     loop_state = pickle.load(open(loop_state_path, 'rb'))
    #
    #     H = pd.read_csv(history_path)
    #     H = {col: list(H[col].values) for col in H.columns}
    #
    #     # HACK: (TODO: Think how to do it nicely)
    #     # TODO(kudkudak): Test!
    #     os.system("cp " + os.path.join(save_path, "history.pkl") + " " + os.path.join(save_path, "history.pkl.bckp"))
    #     epoch_start = loop_state['last_epoch_done_id'] + 1
    #     def reload_pickled_history(epoch, logs):
    #         logger.info("reload_pickled_history({})".format(epoch))
    #         if epoch == epoch_start:
    #             assert len(model.history.history) == 0
    #             logger.info("Loading pickled history")
    #             H_pickle = pickle.load(open(os.path.join(save_path, "history.pkl")))
    #             setattr(model.history, "history", H_pickle)
    #             logger.info("Done loading pickled history")
    #             assert len(model.history.history) != 0
    #     # WARNING: After train_begin
    #     callbacks.insert(0, LambdaCallbackPickable(on_epoch_begin=reload_pickled_history))
    #
    #     # load all callbacks from loop_state
    #     for e, e_loaded in zip(custom_callbacks, loop_state['callbacks']):
    #         assert type(e) == type(e_loaded)
    #         if hasattr(e, "__setstate__"):
    #             e.__setstate__(e_loaded.__dict__)
    #         else:
    #             e.__dict__.update(e_loaded.__dict__)
    # else:
    #     logger.info("Removing " + history_path)
    #     os.system("rm " + history_path)
    #     H = {}
    #     model.save_weights(os.path.join(save_path, "init_weights.h5"))
    #     # A  bit ugly, but last_epoch is here -1 epoch
    #     model.save_weights(os.path.join(save_path, "model_last_epoch.h5"))



    # TODO: Fix
    # def lr_schedule(epoch, logs):
    #     for e, v in learning_rate_schedule:
    #         if epoch >= e:
    #             model.optimizer.lr.set_value(v)
    #             break
    #     logger.info("Fix learning rate to {}".format(v))
    #
    # callbacks.append(LambdaCallback(on_epoch_end=lr_schedule))


    callbacks.append(LambdaCallback(on_epoch_end=partial(_save_and_print_history, save_path=save_path)))
    callbacks.append(ModelCheckpoint(monitor=checkpoint_monitor,
        model=model,
        optimizer=model.optimizer,
        save_best_only=True,
        mode='max',
        filepath=os.path.join(save_path, "model_best_val.pt")))
    # TODO: Tego nie savuje..
    callbacks.append(ModelCheckpoint(monitor=checkpoint_monitor,
        model=model,
        optimizer=model.optimizer,
        save_best_only=False,
        mode='max',
        filepath=os.path.join(save_path, "model_last_epoch.pt")))
    callbacks.append(LambdaCallback(on_epoch_end=partial(_save_loop_state, save_callbacks=custom_callbacks,
        save_path=save_path)))

    _ = model.fit_generator(train,
        initial_epoch=loop_state['last_epoch_done_id'] + 1,
        steps_per_epoch=steps_per_epoch,
        epochs=n_epochs,
        verbose=1,
        valid_generator=valid,
        callbacks=callbacks)
