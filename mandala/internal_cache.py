import os
import json

from mandala import MANDALA_CACHE

# TODO: change internal cache to single file!
# TODO: use sth else than json?


def cached_value_to_index(value):
    cache = _load_internal_cache()
    if value in cache:
        output_index = cache.index(value)
    else:
        output_index = _add_to_internal_cache(value)

    return output_index

def _load_internal_cache():
    cache = []
    for i in xrange(len(os.listdir(MANDALA_CACHE))):
        file_name = str(i) + '.json'
        with open(os.path.join(MANDALA_CACHE, file_name), 'r') as f:
            cache.append(json.load(f))

    return cache


def _add_to_internal_cache(value):
    output_index = len(os.listdir(MANDALA_CACHE))
    new_filename = str(output_index) + '.json'
    with open(os.path.join(MANDALA_CACHE, new_filename), 'w') as f:
        json.dump(value, f)
    return output_index