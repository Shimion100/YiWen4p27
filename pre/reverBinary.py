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

def transposeImage(filename):
	img=Image.open(filename);
	img=img.transpose(Image.FLIP_LEFT_RIGHT);
	img=img.resize((28,28),Image.ANTIALIAS);
	img.save(filename,"JPEG");

def preOp(fileName):
    print "Processing ... ", fileName
    if fileName == 'temp.jpg':
        return
    #only process the jpg file
#    sufix = os.path.splitext(fileName)[1][1:]
#    if sufix != 'jpg':
#        return

    #load background
    backGround = Image.open('./temp.jpg')
    backGround = backGround.convert('RGBA')

    print fileName;    
    #open the image
    img = Image.open(fileName)
    img = img.convert("RGBA")
   
    pixdata = img.load()
    rows=img.size[0];
    cols=img.size[1];
    #scan by cols
    for y in xrange(cols):
        for x in xrange(rows):
            if pixdata[x, y][0] < 90:
                pixdata[x, y] = (255,255,255, 255)
	    else:
		pixdata[x,y]=(0,0,0,0)
 
    for y in xrange(cols):
        for x in xrange(rows):
            if pixdata[x, y][1] < 90:
                pixdata[x, y] = (255, 255, 255, 255)
	    else:
                pixdata[x, y] = (0, 0, 0, 0)

    for y in xrange(cols):
        for x in xrange(rows):
            if pixdata[x, y][2] < 90:
		pixdata[x,y]=(255,255,255,255)
	    else:
                pixdata[x, y] = (0, 0, 0, 0)
  
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
    
def binaryzationJpg(src):
        i=1;
        print "start binaryzationJpg()"
        
	files=subfilesName(src+"/data100");
        length=len(files);
        print "###########",src,"##",length,"###########"
        i=1;
        for f in files:
                if os.path.isfile(os.path.join(src+"/data100",f)):
                        preOp(os.path.join(src+"/data100",f));
                        i=i+1;
			shutil.copy(os.path.join(src+"/data100",f),os.path.join(src+"/data28",f))
			shutil.copy(os.path.join(src+"/data100",f),os.path.join(src+"/data28",f+"_ts"))
			cropImage(os.path.join(src+"/data28",f));
			transposeImage(os.path.join(src+"/data28",f+"_ts"))
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
