# -*- coding: utf-8 -*-
"""
Simple data getters. Each returns iterator for train and dataset for test/valid
"""

from pytoune.framework.callbacks import ModelCheckpoint, Callback

# Might misbehave with tensorflow-gpu, make sure u use tensorflow-cpu if using Theano for keras
try:
    import tensorflow
except:
    pass

import pandas as pd

import os
import pickle
import logging
logger = logging.getLogger(__name__)

from src.callbacks import LambdaCallback, DumpTensorflowSummaries

def training_loop(model, train, valid, n_epochs, save_path, steps_per_epoch):

    # TODO: Add back reloading
    # if os.path.exists(os.path.join(save_path, "loop_state.pkl")):
    #     logger.info("Reloading loop state")
    #     loop_state = pickle.load(open(os.path.join(save_path, "loop_state.pkl")))
    # else:

    loop_state = {'last_epoch_done_id': -1}

    if os.path.exists(os.path.join(save_path, "model.h5")):
        model.load_weights(os.path.join(save_path, "model.h5"))

    callbacks = []

    # TODO: Fix
    # def lr_schedule(epoch, logs):
    #     for e, v in learning_rate_schedule:
    #         if epoch >= e:
    #             model.optimizer.lr.set_value(v)
    #             break
    #     logger.info("Fix learning rate to {}".format(v))
    #
    # callbacks.append(LambdaCallback(on_epoch_end=lr_schedule))

    def save_and_print_history(epoch, logs):
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

    callbacks.append(LambdaCallback(on_epoch_end=save_and_print_history))
    callbacks.append(ModelCheckpoint(monitor='val_acc', filename=os.path.join(save_path, "model.h5")))

    def save_loop_state(epoch, logs):
        loop_state = {"last_epoch_done_id": epoch}
        pickle.dump(loop_state, open(os.path.join(save_path, "loop_state.pkl"), "wb"))
    callbacks.append(LambdaCallback(on_epoch_end=save_loop_state))

    _ = model.fit_generator(train,
        initial_epoch=loop_state['last_epoch_done_id'] + 1,
        steps_per_epoch=steps_per_epoch,
        epochs=n_epochs,
        verbose=1,
        valid_generator=valid,
        callbacks=callbacks)
