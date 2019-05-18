import pandas
import os
import re
def read_metrics_as_df(file, old_name, new_name):
    df = pandas.read_csv(file)
    df=df.rename(index=str,columns={'Value':new_name})
    return df

def get_all_metrics(dir):
    file_lst = os.listdir(dir)
    file_lst = list(filter(lambda x: re.findall(r'\.csv$',x), file_lst))
    return file_lst

def generate(dir):
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

    config_batch4_dir = './batch4lr0.004'
    res_file = os.path.join(config_batch4_dir,'result.csv')
    if os.path.isfile(res_file):
        os.remove(res_file)
    df_batch4 = generate(config_batch4_dir)
    df_batch4.iloc[::5, :].round(2).to_csv(res_file)
    


    config_batch8_lr0_004_dir = './batch8lr0.004'
    res_file = os.path.join(config_batch8_lr0_004_dir,'result.csv')
    if os.path.isfile(res_file):
        os.remove(res_file)
    df_batch8_lr0_004 = generate(config_batch8_lr0_004_dir)
    df_batch8_lr0_004.iloc[::5, :].round(2).to_csv(res_file)