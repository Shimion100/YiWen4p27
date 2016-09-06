import cPickle, gzip, numpy

# Load the dataset
f = gzip.open('file.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
print train_set,valid_set,test_set
f.close()
