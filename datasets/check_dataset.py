from utils.misc import loadHDF5
import glob,os
for f in glob.glob('./synthetic/*.h5'):
    print os.path.basename(f)
    dset = loadHDF5(f)
    for k in dset:
        print k, dset[k].max(), dset[k].min(), dset[k].mean()
