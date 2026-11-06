"""Lossless float parsing for frozen winner-model feature interchange."""
import pandas as pd


def read_csv(path, **kwargs):
    if kwargs.pop("float_precision", "round_trip") != "round_trip":
        raise ValueError("Frozen winner features require lossless round-trip float parsing")
    return pd.read_csv(path,float_precision="round_trip",**kwargs)
