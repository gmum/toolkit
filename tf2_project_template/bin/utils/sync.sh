#!/usr/bin/env bash

if [ "$1" = "uj" ]; then
	WHERE=jastrzebski@access.capdnet.ii.uj.edu.pl:/home/jastrzebski/$PNAME
	rsync -vrpa * --exclude-from=bin/utils/exclude.rsync $WHERE
fi