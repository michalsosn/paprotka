#!/bin/sh

virtualenv venv
./venv/bin/pip3 install -e ..
./venv/bin/pip3 install jupyter
./venv/bin/jupyter notebook
