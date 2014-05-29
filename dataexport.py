#Dump each file to a flattened array-feature vector is all files interleaved
import glob
import os 

import numpy as np
from scipy.io import netcdf as ncd

def extract_data(filename, var_name):
	fp = ncd.netcdf_file(os.path.join('../data',filename))
	d = fp.variables[var_name][:,:,:]
	fp.close()
	return d

if __name__ == '__main__':
	drange = range(1985, 2010)	
	VAR = "TMP_2maboveground"
	DIM = (1, 190, 384)

	fstr = "TMP_*{}f{:02}.nc"
	astr = "TMP_*_a.nc"
	dates = sorted([(y,m) for m in range(1,13) for  y in drange])
	fnames = ["TMP_{0}{1:02d}".format(y, m) for y,m in dates]
        dstamp= [y*100+m for y, m in dates]
	with open("dataflat.txt", 'a') as df:	
		all_data = []
		for fn, ds in zip(fnames, dstamp):
			print fn
			data = []
 			for i in range(1, 10):
				filename = "{0}_f{1:02d}.nc".format(fn, i)
				d = extract_data(filename, VAR)
				data.append(d.flatten())
			filename = "{0}_a.nc".format(fn)
			d = extract_data(filename, VAR)
			data.append(d.flatten())
			all_data.append(np.vstack(data).T)
		#np.savetxt(df, np.vstack(all_data).flatten(), fmt='%f4')
	darr = np.vstack(all_data)
	print darr.min(), darr.max()
