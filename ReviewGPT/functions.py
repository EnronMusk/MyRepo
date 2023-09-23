import pandas as pd
import numpy as np
import gzip

def downloadData(path : str, n : int) -> pd.DataFrame:
    return pd.read_json(path, lines=True).head(n)
