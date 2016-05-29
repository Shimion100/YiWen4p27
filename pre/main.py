import os,random,string,shutil,binaryzation
path=".";
src="./src";
#path=raw_input("src path:");
train=0.0;
verify=0.0;
test=0.0;
#n=raw_input("input train mount decimals:");
#m=raw_input("input verify mount decimals:");
#l=raw_input("input test mout decimals:");
def rename():
	for f in files:
	    if os.path.isfile(os.path.join(path,f))==True:
		i=i+1;
		name='{0}_{1:0{2}d}'.format(prefix,i,4);
		os.rename(os.path.join(path,f),os.path.join(path,name));
		print f,"-->",name

# get sub dirs
def subdirs(path):
    dl = [];
    for i in os.walk(path, False):
        for d in i[1]:
            dl.append(os.path.join(path, d))
    return dl

#get sub file
def subfiles(path):
    fl = [];
    for i in os.walk(path, False):
        for f in i[2]:
            fl.append(os.path.join(path, f))
    return fl

def subfilesName(path):
    fl = [];
    for i in os.walk(path, False):
        for f in i[2]:
            fl.append(f)
    return fl


def random_str(randomlength=8):
    str = ''
    chars = 'AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789'
    length = len(chars) - 1
    random = Random()
    for i in range(randomlength):
        str+=chars[random.randint(0, length)]
    return str

def random_string(randomlength=8):
    a = list(string.ascii_letters)
    random.shuffle(a)
    return ''.join(a[:randomlength])

def separationFile():
	print "input the train verify test number!"
	train=input();
	verify=input();
	test=input();
	train=train*0.01;
	verify=verify*0.01;
	test=test*0.01;
	#train=train*0.01;
	#verify=verify*0.01;
	#test=test*0.01;
	print "this is the factor:",train,",",verify,",",test
	files=os.listdir(src);
	dirs=subdirs(src);
	i=1;
	for d in dirs:
		files=subfilesName(d);
		length=len(files);
		print "###########",d,"###########"
		i=1;
		for f in files:
			name=random_string(8)+"_"+f;
			if i<=length*train:
				p=path+"/train";
			elif i<=(length*train+length*verify):
				p=path+"/verify";
			else:
				p=path+"/test";
			shutil.copy(os.path.join(d,f),os.path.join(p,name));
#			os.rename(os.path(join(d,f),os.path.join(p,name);
			i=i+1;
			print i,":",d," file:",f,"--->",p," file:",name
def binaryzationJpg(src):
        dirs=subdirs(src);
        i=1;
        for d in dirs:
                files=subfilesName(d);
                length=len(files);
                print "###########",d,"###########"
                i=1;
                for f in files:
			if os.path.isfile(os.path.join(d,f)):
                       		 preOp(os.path.join(d,f));
			i=i+1;
                        print i,":",f,"is finish..."
def binaryzations():
	binaryzationJpg("./train");
	binaryzationJpg("./verify");
	binaryzationJpg("./test");
#command=raw_input
binaryzation();
print "the end"
