#!/bin/bash
uv venv .venv --seed --python 3.12
uv pip install --upgrade pip
uv pip install setuptools wheel
uv pip install --editable . -vv
