import pandas as pd 
import numpy as np
import pdb

seq_dir = '/lab_data/behrmannlab/vlad/docnet/optseq'

runs = 8

for rr in range(1,9):
    df= pd.read_csv(f'{seq_dir}/docnet_mvpa-00{rr}.par', header=None, delim_whitespace=True)
    df = df.replace(np.nan, 'fix')
    cols = ['onset', 'index', 'stim_duration','ones', 'stim']
    df.columns = cols

    df.to_csv(f'{seq_dir}/seperate_fix/trials_{rr}.csv', columns = ['stim_duration', 'stim'], index=False)
    final_csv = pd.DataFrame(columns = ['onset','stim_duration', 'stim'])
    for sd in range(0, len(df)):
        #pdb.set_trace()
        if df.iloc[sd, 4] != 'fix':
            final_csv = final_csv.append(pd.Series([df.iloc[sd,0]+8,df.iloc[sd+1,2], df.iloc[sd,4]], index = final_csv.columns),ignore_index=True)
    
    final_csv.to_csv(f'{seq_dir}/integrated_fix/trials_{rr}.csv', columns = ['onset', 'stim_duration', 'stim'], index=False)
        
        
    #final_csv['stim_duration'] =  df['stim_duration']
    #final_csv['stim'] =  df['stim']

    