from PIL import Image
from numpy import genfromtxt
import gzip, cPickle
from glob import glob
import numpy as np
import os
import theano
class pcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    Blue='\34[95m'
    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''


def dir_to_dataset(glob_files):
    print("Gonna process:\n\t %s"%glob_files)
    dataset = []
    for file_count, file_name in enumerate( sorted(glob(glob_files),key=len) ):
        image = Image.open(file_name)
        print file_name
        img = Image.open(file_name).convert('LA') #tograyscale
        pixels = [f[0] for f in list(img.getdata())]
        dataset.append(pixels)
        if file_count % 1000 == 0:
            print("\t %s files processed"%file_count)
    return np.array(dataset)

def initDataSet():
    path="./train/data100/*";
    Data= dir_to_dataset(path)
    # Data and labels are read
    #set a filename
    afileName = str(os.getcwd()) + "\\kmeans.data"
    #create the file
    file = open(afileName, 'w')

    #put the data into a file.
    print len(Data)
    for x in xrange(0, len(Data)):
        aRow = Data[x]
        for pix in xrange(0, len(aRow)):
            file.write(str(aRow[pix]))
            if pix != len(aRow) - 1:
                file.write(",")
        print len(aRow)
        file.write("\n")
    file.close()

#run
initDataSet()
