#!/bin/bash

set -e

export LANG=C.UTF-8

apt-get update
apt-get install portaudio19-dev

pip install --no-cache-dir glob2 Cython
pip install --no-cache-dir -e ..
jupyter nbextension install --py --user widgetsnbextension
jupyter nbextension enable --py widgetsnbextension
