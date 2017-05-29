#!/bin/sh

virtualenv venv
./venv/bin/pip3 install --no-cache-dir -e .. 
./venv/bin/pip3 install --no-cache-dir jupyter
./venv/bin/jupyter notebook "$@"
