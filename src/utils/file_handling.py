import os
import pickle


def save_pickle(pthing, fname):
	with open('pickles/' + fname + '.pickle', 'wb') as f:
		pickle.dump(pthing, f, pickle.HIGHEST_PROTOCOL)
	return

def open_pickle(fname):
	with open('pickles/' + fname + '.pickle', 'rb') as f:
		d = pickle.load(f)
	return d

def open_file(poem):
	if '.' not in poem:
		poem += '.txt'
	with open(os.path.join('input', poem)) as f:
		return f.read()
