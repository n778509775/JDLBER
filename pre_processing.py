# !/usr/bin/env python
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale, minmax_scale, Imputer

def read_csv(filename, take_log):
	data_df = pd.read_csv(filename)
	dataset = {}
	dataset['labels'] = data_df.iloc[:,0].tolist()
	dataset['board'] = data_df.iloc[:,1].tolist()
	mz_exp = np.transpose(np.array(data_df.iloc[:,2:]))

	#remove NaN
	if np.any(np.isnan(mz_exp)):
		my_imputer = Imputer()
		mz_exp = my_imputer.fit_transform(mz_exp)

	if take_log:
		mz_exp = np.log2(mz_exp + 1)
	dataset['mz_exp'] = mz_exp
	dataset['feature'] = data_df.columns.values.tolist()[2:]
	return dataset

def pre_processing(dataset_file_list, pre_process_paras):
    # parameters
    take_log = pre_process_paras['take_log']
    standardization = pre_process_paras['standardization']
    scaling = pre_process_paras['scaling']

    dataset_list = []
    for data_file in dataset_file_list:
        dataset = read_csv(data_file, take_log)
        if standardization:
            scale(dataset['mz_exp'], axis=1, with_mean=True, with_std=True, copy=False)
        if scaling:  # scale to [0,1]
            minmax_scale(dataset['mz_exp'], feature_range=(0, 1), axis=1, copy=False)
        dataset_list.append(dataset)

    return dataset_list


