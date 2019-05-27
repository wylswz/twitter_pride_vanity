"""
Maintainer: Yunlu Wen <yunluw@student.unimelb.edu.au>

This is a script of processing evaluation metrics of COCO object detection evaluation. It only supports csv from tensorboard
of tensorflow object_detection project
"""

import pandas
import os
import re

def read_metrics_as_df(file, old_name, new_name):
    """
    Read one metric file as dataframe, Assign a new column name to "Value" col
    :param file:
    :param old_name:
    :param new_name:
    :return:
    """
    df = pandas.read_csv(file)
    df=df.rename(index=str,columns={'Value':new_name})
    return df

def get_all_metrics(dir):
    """
    List all metric files in that directory
    :param dir:
    :return:
    """
    file_lst = os.listdir(dir)
    file_lst = list(filter(lambda x: re.findall(r'\.csv$',x), file_lst))
    return file_lst

def generate(dir):
    """
    Combine methods above to generate a complete metrics sheet
    :param dir:
    :return:
    """
    metric_files = get_all_metrics(dir)
    dfs = []
    for m in metric_files:
        metric_name = m.replace('.csv', '').split('-')[-1]
        full_path = os.path.join(dir, m)
        dfs.append(read_metrics_as_df(full_path,'Value',metric_name).set_index('Step').drop(columns=['Wall time']))
    base_df = dfs[0]
    for df in dfs[1:]:
        base_df = base_df.join(df, on='Step')
    return base_df

if __name__ == "__main__":

    sample_seed = 544

    eval_dir = './batch8lr0.004decay0.8'
    res_file = os.path.join(eval_dir, 'result.csv')
    if os.path.isfile(res_file):
        os.remove(res_file)
    df = generate(eval_dir)
    df.iloc[::5, :].round(2).to_csv(res_file)
    

