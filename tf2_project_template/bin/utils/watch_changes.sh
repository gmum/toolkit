#!/usr/bin/env bash

# CONFIGURATION
MACHINE=uj

watchman watch-del-all
pkill watchman
watchman-make -p 'experiments/*json' 'experiments/*.py' 'experiments/**/*.py' 'experiments/**/*.sh' 'experiments/**/*.gin' 'src/**/*.py' 'bin/**/*.py' 'bin/*.py' 'src/*.py' 'configs/*gin' --run "bash `pwd`/bin/utils/sync.sh $MACHINE" &
