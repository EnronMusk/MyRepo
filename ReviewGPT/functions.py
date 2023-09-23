import pandas as pd
import numpy as np
import gzip

def _parse(path : str):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def _getDF(path : str):
  i = 0
  df = {}
  for d in _parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

"""
@params 
n is number of observations
path is path

"""
def downloadData(path : str, n : int) -> pd.DataFrame:
    return pd.read_json(path).head(n)

def say():
  print("HI XD")