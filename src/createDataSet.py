from PIL import Image
from xml.dom import minidom
from pandas import Series
from numpy import genfromtxt
# import gzip, cPickle for this is used simple
import gzip
#import _pickle as cPickle
#
# for nickname cPickle because pickle has replase cPickle in pandans
import pickle as cPickle
import pickle
from glob import glob
import numpy as np
import os
#import theano
import xml.etree.ElementTree
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
        print(file_name)
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
    print(len(Data))
    xmlFile=str(os.getcwd())+"\\image.xml"
    if(os._exists(xmlFile)==False):
         document=minidom.Document()
         document.appendChild(document.createComment("this is used for save a image file"))
         imagelist=document.createElement("Images")
         document.appendChild(imagelist)
         f=open("image.xml","w")
         document.writexml(f,addindent=' '*4, newl='\n', encoding='utf-8')
         f.close()
    root=xml.dom.minidom.parse('image.xml')
    imagesRoot=root.documentElement
    #root=xml.etree.ElementTree.parse("image.xml");

    for x in range(0, len(Data)):
        imageRoot=document.createElement("Image")
        id=document.createElement("Id")
        data=document.createElement("Data")
        aRow = Data[x]
        value=[]
        for pix in range(0, len(aRow)):
            file.write(str(aRow[pix]))
            value.append(int(str(aRow[pix])))
            if pix != len(aRow) - 1:
                file.write(",")
        #print(len(aRow))
        id.appendChild(document.createTextNode("1"))
        data.appendChild(document.createTextNode(str(value)))
        imagesRoot.appendChild(imageRoot)
        imageRoot.appendChild(id)
        imageRoot.appendChild(data)
        file.write("\n")
    f=open("image.xml","w")
    root.writexml(f,addindent=' '*4, newl='\n', encoding='utf-8')
    f.close()

#run
initDataSet()
