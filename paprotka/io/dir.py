import os


def walk_dir(path, key=None):
    for dirpath, _, files in os.walk(path):
        for name in sorted(files, key=key):
            yield os.path.join(dirpath, name)
