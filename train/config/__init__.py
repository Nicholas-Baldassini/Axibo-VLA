"""Training configs: one file per run type, each exposing a CONFIG dict. load_config("openvla") returns config/openvla.py's."""

import importlib


def load_config(name):
    return importlib.import_module(f"config.{name}").CONFIG
