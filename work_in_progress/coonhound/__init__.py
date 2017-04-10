#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import json
import os
import re

def _remove_chars(string, old='/', new='%', meta='_'):
    assert isinstance(old, str)
    assert isinstance(new, str)
    assert isinstance(meta, str)
    assert len(set([old, new, meta])) == 3
    assert map(len, [old, new, meta]) == [1, 1, 1]
    return re.sub(re.escape(old), meta+new, re.sub(re.escape(meta), meta+meta, string))

# invite Staszek to find better names
def _basic_obj_to_dirname(obj):
    return _remove_chars(json.dumps(obj, separators=(',', ':'), sort_keys=True), old='/', new='%', meta='_')


class DirectoryHound():

    def __init__(self, root_dirname):
        self._root_dirname = root_dirname

    def find(self, **kwargs):
        print self._root_dirname
        print [_basic_obj_to_dirname(item) for item in sorted(kwargs.items())]
        return os.path.join(self._root_dirname, *[_basic_obj_to_dirname(item) for item in sorted(kwargs.items())])
