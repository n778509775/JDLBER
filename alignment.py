from scipy.signal import find_peaks,peak_widths
import time
import pyopenms
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--data_path', type=str, default='data/')
	parser.add_argument('--target', type=str, default='yangnr2.csv')
	parser.add_argument('--source', type=str, default='yangnr1.csv')

	config = parser.parse_args()
	print(config)
	
data_path = config.data_path
def read_csv_faster(filename):
	data_df = pd.read_csv(data_path + filename,index_col=0)
	dataset = {}
	dataset['labels'] = data_df.iloc[:,0].tolist()
	dataset['board'] = data_df.iloc[:,1].tolist()
	dataset['mz_exp'] = np.transpose(np.array(data_df.iloc[:,2:]))
	dataset['feature'] = data_df.columns.values.tolist()[2:]
	return dataset

target = read_csv_faster(config.target)
source = read_csv_faster(config.source)

super_spectrum_mz = []
for i in target['feature']:
	super_spectrum_mz.append(float(i))

mz = []
for i in source['feature']:
	mz.append(float(i))

spectrums=[]
based_intensity = np.transpose(source['mz_exp'])
for i in range(based_intensity.shape[0]):
	peak_intensity=based_intensity[i]
	spectrum=pyopenms.MSSpectrum()
	spectrum.set_peaks([mz,peak_intensity])
	spectrum.sortByPosition()
	spectrums.append(spectrum)

super_spectrum_mz=np.array(super_spectrum_mz)
print(super_spectrum_mz.shape)		

aligned_intensities=[]
aligner=pyopenms.SpectrumAlignment()
target_spectrum=pyopenms.MSSpectrum()
target_spectrum.set_peaks([super_spectrum_mz,np.zeros_like(super_spectrum_mz)])
target_spectrum.sortByPosition()

for spectrum in spectrums:
	alignment=[]
	aligner.getSpectrumAlignment(alignment,spectrum,target_spectrum)	
	_,to_align_intensity=spectrum.get_peaks()
	aligned_intensity=np.zeros_like(super_spectrum_mz)
	for t in alignment:
		aligned_intensity[t[1]]=to_align_intensity[t[0]]
	aligned_intensities.append(aligned_intensity)

aligned_intensities=np.array(aligned_intensities)
out_df=pd.DataFrame()
out_df["labels"]=source['labels']
out_df["board"]=source['board']
for i,mz in enumerate(super_spectrum_mz):
	out_df[mz]=aligned_intensities[:,i]
out_df.to_csv("aligned_file.csv",index=False)

