#!/bin/bash

set -e

virtualenv venv
source venv/bin/activate
pip3 install --no-cache-dir glob2 Cython
pip3 install --no-cache-dir -e ..
jupyter nbextension install --py --user widgetsnbextension
jupyter nbextension enable --py widgetsnbextension
jupyter notebook "$@"
