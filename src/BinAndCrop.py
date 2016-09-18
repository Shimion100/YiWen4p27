import os
import sys
from PIL import Image
#from PIL.Image import core as image
import os, random, string, shutil

path = "."
src = "./src"

"""
    walk through a directory and get all the file in this directory.
"""


def subfilesName(path):
    fl = []
    for i in os.walk(path, False):
        for f in i[2]:
            fl.append(f)
    return fl


"""
    crop the image to fixed size
"""


def cropImage(fileName):
    print("Processing ... ", fileName)
    # open the image
    img = Image.open(fileName)
    img = img.resize((28, 28), Image.ANTIALIAS)
    img.save(fileName, "JPEG")


"""
    PreOperation of Image, include the binarization and crop
"""


def bin(fileName):
    print("Processing ... ", fileName)
    # open the image
    img = Image.open(fileName)
    img = img.convert("L")

    pixdata = img.load()
    rows = img.size[0]
    cols = img.size[1]
    # scan by cols

    # you could change the range function to xrange function
    for y in range(cols):
        for x in range(rows):
            if pixdata[x, y] > 127:
                pixdata[x, y] = 255
            else:
                pixdata[x, y] = 0
    img.save(fileName, "JPEG")


def binaryzationJpg(src):

    print("start binaryzationJpg()")
    files = subfilesName(src)
    length = len(files)
    print("###########", src, "##", length, "###########")
    i = 1
    for f in files:
        if os.path.isfile(os.path.join(src , f)):
            bin(os.path.join(src , f))
            i += 1
            cropImage(os.path.join(src , f))


"""
    only process one directory.
"""
def binaryzations():
    binaryzationJpg("./train/test")


# command=raw_input
binaryzations()

