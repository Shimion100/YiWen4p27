import os
import sys
import Image
import os,random,string,shutil
path=".";
src="./src";

def subfilesName(path):
    fl = [];
    for i in os.walk(path, False):
        for f in i[2]:
            fl.append(f)
    return fl

def cropImage(fileName):
    print "Processing ... ", fileName
    #open the image

    img = Image.open(fileName)
    img=img.resize((28,28),Image.ANTIALIAS);    
    img.save(fileName, "JPEG")

def preOp28(fileName):
    print "Processing ... ", fileName
    if fileName == 'temp.jpg':
        return
    #only process the jpg file
#    sufix = os.path.splitext(fileName)[1][1:]
#    if sufix != 'jpg':
#        return

    #load background
    backGround = Image.open('./temp28.jpg')
    backGround = backGround.convert('RGBA')

    print fileName;    
    #open the image
    img = Image.open(fileName)
    img = img.convert("L")
   
    pixdata = img.load()
    rows=img.size[0];
    cols=img.size[1];
    #crop the image to 100 * 100
    if rows != 28 and cols != 28:
        box = (0,0, rows,cols)
        region = img.crop(box)
        positioin = ((28-rows)/2, (28-cols)/2, (28-rows)/2+rows, (28-cols)/2+cols )
        print positioin
        backGround.paste(region,positioin)
        img = backGround
    print "......#####",fileName,"######....."
    
    img.save(fileName, "JPEG")
    img = Image.open(fileName)
    img = img.convert("L")
   
    pixdata = img.load()

    rows=img.size[0];
    cols=img.size[1];
    #scan by cols
    for y in xrange(cols):
        for x in xrange(rows):
            pixdata[x,y]=255  if pixdata[x,y]>=128 else 0

   
    img.save(fileName, "JPEG")


def preOp100(fileName):
    print "Processing ... ", fileName
    if fileName == 'temp.jpg':
        return
    #only process the jpg file
#    sufix = os.path.splitext(fileName)[1][1:]
#    if sufix != 'jpg':
#        return

    #load background
    backGround = Image.open('./temp100.jpg')
    backGround = backGround.convert('RGBA')

    print fileName;    
    #open the image
    img = Image.open(fileName)
    img = img.convert("L")
   
    pixdata = img.load()
    rows=img.size[0];
    cols=img.size[1];
  
    #crop the image to 100 * 100
    if rows != 100 and cols != 100:
        box = (0,0, rows,cols)
        region = img.crop(box)
        positioin = ((100-rows)/2, (100-cols)/2, (100-rows)/2+rows, (100-cols)/2+cols )
        print positioin
        backGround.paste(region,positioin)
        img = backGround
    print "......#####",fileName,"######....."
   
    img.save(fileName, "JPEG")
    img = Image.open(fileName)
    img = img.convert("L")
   
    pixdata = img.load()
    rows=img.size[0];
    cols=img.size[1];
    #scan by cols
    for y in xrange(cols):
        for x in xrange(rows):
            pixdata[x,y]=255  if pixdata[x,y]>=128 else 0


    img.save(fileName, "JPEG")
    
def binaryzationJpg(src):
        i=1;
        print "start binaryzationJpg()"
        
	files=subfilesName(src+"/data100");
        length=len(files);
        print "###########",src,"##",length,"###########"
        i=1;
        for f in files:
                if os.path.isfile(os.path.join(src+"/data100",f)):
                        preOp100(os.path.join(src+"/data100",f));
                        i=i+1;
		        cropImage(os.path.join(src+"/data28",f));
			preOp28(os.path.join(src+"/data28",f));
	               
def binaryzations():
        binaryzationJpg("./train");
        binaryzationJpg("./verify");
        binaryzationJpg("./test");
#command=raw_input
binaryzations();




#if __name__ == "__main__":
#    root = "./"
#    for i in os.listdir(root):
#        if os.path.isfile(os.path.join(root,i)):
#            preOp(i) 
