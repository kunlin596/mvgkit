#!/usr/bin/env bash

REPO_PATH=$(cd -- $(dirname -- $0) && pwd)
export PYTHONPATH=$PYTHONPATH:$REPO_PATH/python
export PATH=$PATH:$REPO_PATH/scripts:$REPO_PATH/apps
# echo exported PYTHONPATH=$PYTHONPATH
