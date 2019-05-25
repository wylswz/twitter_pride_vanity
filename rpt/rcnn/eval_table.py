import json
import os
import re
from pandas import DataFrame as DF

def filename_to_colname(filename):
    name_secs = filename.split('_')
    tab_col_name = name_secs[-2:]
    tab_col_name = ''.join(tab_col_name).replace('.json', '')
    return tab_col_name


def generate(path) -> DF:
    files = os.listdir(path)
    
    dfs = []
    for f in files:
        if re.findall(r'\.json$', f):
            tab_col_name = filename_to_colname(f)
            with open(f, 'r') as fp:
                data = json.load(fp)
                metric = [round(d[2],4) for d in data]
                step = [d[1] for d in data]
                df = DF(data=metric,index=step, columns=[tab_col_name])
                dfs.append(df)
    
    first = DF(dfs[0])
    newdf = first.join(dfs[1:], how='left')
    return newdf

if __name__ == "__main__":
    newdf = generate(".")
    newdf.to_csv("metrics.csv")