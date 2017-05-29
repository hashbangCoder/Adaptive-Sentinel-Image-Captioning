
import torchvision.datasets as dset
import torch, os
from nltk import word_tokenize
import numpy as np
import cPickle as pickle

dl = dset.CocoCaptions(root='../../Datasets/COCOData/Data/train2014/',annFile = '../../Datasets/COCOData/Data/annotations/captions_train2014.json')

class Vocabulary(object):		#'../FormattedData/Multi-LSTM/vocabulary_fg.txt'
	def __init__(self):
		self.word2idx = {}
		self.idx2word= {}
		self.OOV = 0
		#if not os.path.isfile('vocab.pkl'):
		self.getVocab()
		
		#self.getPretrainedEmbeds()

	def getVocab(self):
		print 'Creating vocabulary dictionary...'
		idx = 1
		for sample in dl.coco.anns.values():
			for word in map(lambda x: x.lower().strip(),word_tokenize(sample['caption'])):
				if word not in self.word2idx:
					self.word2idx[word] = idx 
					self.idx2word[idx] = word
					idx+=1

		self.word2idx['<unk>'] = len(self.word2idx)+1
		self.idx2word[len(self.word2idx)+1] = '<unk>'

		self.word2idx['<go>'] = len(self.word2idx)+1
		self.idx2word[len(self.word2idx)+1] = '<go>'

		self.word2idx['<stop>'] = len(self.word2idx)+1
		self.idx2word[len(self.word2idx)+1] = '<stop>'
		
		self.vocabSize = len(self.word2idx)

		with open('vocab.pkl','wb') as f:
			pickle.dump(self.word2idx,f, protocol=pickle.HIGHEST_PROTOCOL)
		with open('reverse_vocab.pkl','wb') as f:
			pickle.dump(self.idx2word,f, protocol=pickle.HIGHEST_PROTOCOL)
		

def getPretrainedEmbeds(vocab,pretrained_path='../../Text_Classification/Pretrained_WordVecs/glove.840B.300d.txt'):		
	if os.path.isfile('embeds.pkl'):
		raise Exception('Pretrained word embeds exist. Delete file and run again...')

	print 'Extracting pretrained embeddings. Can take a few minutes...'
	embeddings = torch.randn(len(vocab)+1,300)
	embeddings[0].zero_()
	cnt = 0

	with open(pretrained_path,'r') as f:
		for line in f:
			values = line.split()
			word = values[0].lower().strip()
			if word in vocab:
				cnt += 1
				coefs = np.asarray(values[1:], dtype='float32')
				embeddings[vocab[word]] = torch.from_numpy(coefs)

	print 'Total number of words with pretrained embeddings : ', cnt
	with open('embeds.pkl','wb') as f:
		pickle.dump(embeddings,f, protocol=pickle.HIGHEST_PROTOCOL)
	print 'Saved gLoVe embeddings'


