#!/usr/bin/env python3

from setuptools import setup, find_packages, Extension
from glob import iglob
from os import path


def find_ext_modules(root='.', extension="", dir_prefix='.'):
    def file_into_package(file):
        file, _ = path.splitext(file)
        rel = path.relpath(file, dir_prefix)
        return rel.replace(path.sep, '.')

    expression = '{}/**/*.{}'.format(root, extension)
    return [Extension(file_into_package(file), [file]) for file in iglob(expression)]


LEARN_REQUIRES = ['numpy', 'scipy', 'scikit-learn', 'fastdtw', 'hmmlearn']
VISUALIZATION_REQUIRES = ['matplotlib', 'ipywidgets']
SYSTEM_REQUIRES = ['pyaudio']
INSTALL_REQUIRES = LEARN_REQUIRES + VISUALIZATION_REQUIRES + SYSTEM_REQUIRES

setup(
    packages=find_packages(exclude=['tests/*']),
    ext_modules=find_ext_modules('paprotka', 'pyx'),
    install_requires=INSTALL_REQUIRES,
    setup_requires=['setuptools>=18.0', 'Cython', 'pytest-runner'],
    tests_require=['pytest']
)
