import os
path="./a"
prefix="a"
path=raw_input("input path:");
prefix=raw_input("input prefix name:");
files=os.listdir(path);
i=0;
for f in files:
	if os.path.isfile(os.path.join(path,f))==True:
		i=i+1;
		name='{0}_{1:0{2}d}'.format(prefix,i,4);
		os.rename(os.path.join(path,f),os.path.join(path,name));
		print f,"-->",name
