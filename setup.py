#!/usr/bin/env python3

import glob2 as glob
import os
from setuptools import setup, find_packages, Extension


def find_ext_modules(root='.', extension='', dir_prefix='.'):
    def file_into_package(file):
        file, _ = os.path.splitext(file)
        rel = os.path.relpath(file, dir_prefix)
        return rel.replace(os.path.sep, '.')

    expression = '{}/**/*.{}'.format(root, extension)
    return [Extension(file_into_package(file), [file]) for file in glob.iglob(expression)]


LEARN_REQUIRES = ['numpy', 'scipy', 'scikit-learn', 'fastdtw', 'hmmlearn', 'python_speech_features']
VISUALIZATION_REQUIRES = ['matplotlib', 'jupyter', 'ipywidgets', 'jupyter_contrib_nbextensions']
SYSTEM_REQUIRES = ['pyaudio']
INSTALL_REQUIRES = LEARN_REQUIRES + VISUALIZATION_REQUIRES + SYSTEM_REQUIRES

setup(
    packages=find_packages(exclude=['tests/*']),
    ext_modules=find_ext_modules('paprotka', 'pyx'),
    install_requires=INSTALL_REQUIRES,
    setup_requires=['setuptools>=18.0', 'glob2', 'Cython', 'pytest-runner'],
    tests_require=['pytest']
)
