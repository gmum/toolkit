# Tune learning rate experiment

Experiment conceptually is a list of shell jobs. For convenience this can be wrapped using a python script that prepares jobs, analyses the runs, stores configs, etc. 

We ship an example experiment, where we tune LR for the small CNN on Cifar10. Here is the typical workflow:

1. Prepare experiments: `python experiments/tune_lr/main.py prepare`

2. See prepare configs: `ls experiments/tune_lr/large/configs`

3. Run experiments: `bash experiments/tune_lr/large/batch.sh`

4. See runs: `ls $RESULTS_DIR/tune_lr/large`

5. Process experiment results: `python experiments/tune_lr/main.py report`. Bonus for OSX users: To enable plotting in iterm install ``pip install itermplot``, and uncomment the appropriate line in ``e.sh```.

6. Take a look at the main.py source code to understand better the logic.

Note that running a list of shell jobs can be done using a scheduler. This is best if you develop your own
solution for runnning efficiently such a list.