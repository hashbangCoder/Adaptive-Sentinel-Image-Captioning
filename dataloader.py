import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
import os, random,pdb, torch
from nltk import word_tokenize
from PIL import Image
from itertools import groupby


class CocoCaptions(data.Dataset):
    def __init__(self, root, annFile, transform=None, target_transform=None, vocab=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.vocab = vocab

    def __getitem__(self, index):
        coco = self.coco                
        img_id = self.ids[index]
        if not isinstance(index, slice) and isinstance(index, int):
            img_id = [img_id]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        paths = {im_id: os.path.join(self.root, coco.loadImgs(im_id)[0]['file_name']) for im_id in img_id}
        anns = coco.loadAnns(ann_ids)
        anns = [ann for ann in anns if os.path.isfile(paths[ann['image_id']])]
        
        target = [ann['caption'] for ann in anns]  # if os.path.isfile(paths[ann['image_id']])]
        
        num_captions = [len(list(group)) for key, group in groupby([ann['image_id'] for ann in anns])]
    
        imgs = [Image.open(paths[im_id]).convert('RGB') for im_id in img_id if os.path.isfile(paths[im_id])]
        assert imgs != [], 'Empty list of files'
        if self.transform is not None:            
            imgs = [self.transform(im) for im in imgs]
        
        if self.target_transform is not None:
            batchCaptions, batchTargets = self.target_transform(self.vocab, target)
            return torch.stack(imgs, 0), batchCaptions, batchTargets, num_captions
        else:
            return imgs, target

    def __len__(self):
        return len(self.ids)


class dataloaderBundled():
    def __init__(self, batchSize, epochs, vocab):
        self.maxEpochs = epochs
        self.epoch = 1
        self.batchSize = batchSize
        self.iterInd = 0
        self.globalInd = 1
        self.transform = transforms.Compose([transforms.Scale(256), transforms.RandomSizedCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.vocab = vocab
        self.vocabSize = len(vocab)

        self.trainCOCO = CocoCaptions(root='../../Datasets/COCOData/Data/train2014/', annFile='../../Datasets/COCOData/Data/annotations/captions_train2014.json',
                                      transform=self.transform, target_transform=self.caption_preproc, vocab=self.vocab)
                
        self.trainSamples = len(self.trainCOCO)
        # self.trainLoader = iter(torch.utils.data.DataLoader(self.trainCOCO, num_workers=1, batch_size=batchSize, shuffle=True))

        #Load eval set
        self.testCOCO = CocoCaptions(root='../../Datasets/COCOData/Data/val2014/', annFile='../../Datasets/COCOData/Data/annotations/captions_val2014.json', transform=self.transform, 
                                     vocab=self.vocab)
        self.loadEvalBatch()                
        self.stopFlag = False

    def caption_preproc(self, vocab, samples):        
        intTargets, intCaptions = [], []
        maxLen = 0
        for annot in samples:            
            caption = word_tokenize(annot)
            _intCaption = []
            for word in caption:
                try:
                    _intCaption.append(self.vocab[word.lower().strip()])
                except KeyError:
                    _intCaption.append(self.vocab['<unk>'])
                    # self.unkCount += 1
            
            intCaption = [self.vocab['<go>']] + _intCaption
            intTarget = _intCaption + [self.vocab['<stop>']]
            if maxLen < len(intCaption):
                maxLen = len(intCaption)            

            intCaptions.append(intCaption)
            intTargets.append(intTarget)
                            
        padCaptions = [torch.LongTensor(item + [0] * (maxLen - len(item))) for item in intCaptions]        
        padTargets = [torch.LongTensor(item + [0] * (maxLen - len(item))) for item in intTargets]
        batchCaptions = torch.stack(padCaptions, 0)
        batchTargets = torch.stack(padTargets, 0)
        return batchCaptions, batchTargets        

    def getBundledBatch(self, num_samples):
        if num_samples is None:
            num_samples = self.batchSize
        # self.trainLoader.batch_size = num_samples

        if self.epoch > self.maxEpochs:
            print 'Maximum Epoch Limit reached'
            self.stopFlag = True
            return None
        
        # NOTE :  multi-threaded dataloader doesn't work with a custom target_transform. Needs some work
        if self.iterInd + num_samples - 1 > self.trainSamples:
            data = self.trainCOCO[self.iterInd:]
            #data = self.trainLoader[self.iterInd:]
        else:
            data = self.trainCOCO[self.iterInd:self.iterInd + num_samples]
            #data = self.trainLoader[self.iterInd:self.iterInd + num_samples]
        # try:
        #     data = self.trainLoader.next()
        # except StopIteration:
        #     # New epoch
        #     self.trainLoader = iter(torch.utils.data.DataLoader(self.trainCOCO, batch_size=num_samples, num_workers=2, shuffle=True))

        
        self.globalInd += 1
        self.iterInd += num_samples
        if self.iterInd > self.trainSamples:
            self.iterInd = 0            
            self.epoch += 1
            self.globalInd = 1
        else:
            pass                
        return data

    def loadEvalBatch(self, num_samples=3):
        # select 3 samples at random
        random.shuffle(self.testCOCO.ids)
        self.testImgs, self.testCaptions = [], []
        self.testImID = [0]*num_samples
        for i in range(num_samples):
            im, cap = self.testCOCO[i]
            self.testImgs.append(im[0])
            self.testCaptions.append(cap[0])        
        self.testImgs = torch.stack(self.testImgs, 0)        



# class Dataloader():
#     def __init__(self,batchSize,epochs,vocab):
#         self.maxEpochs = epochs
#         self.epoch = 1
#         self.batchSize = batchSize
#         self.iterInd = 0
#         self.globalInd = 1
#         self.transform = transforms.Compose([transforms.Scale(256),transforms.RandomSizedCrop(224),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#         self.vocab = vocab
#         self.vocabSize = len(vocab)

#         self.trainCOCO = dset.CocoCaptions(root='../../COCOData/Data/train2014/',annFile = '../../COCOData/Data/annotations/captions_train2014.json')
#         self.singletrainAnns = self.trainCOCO.coco.getAnnIds()
#         self.singletrainSamples = len(self.singletrainAnns)
#         # self.trainAnns = self.trainCOCO.coco.imgToAnns.items()        
#         # self.trainSamples = len(self.trainAnns)
#         self.trainAnns = self.trainCOCO.coco.getAnnIds()
#         self.trainSamples = len(self.trainAnns)

#         #Load eval set
#         self.testCOCO = dset.CocoCaptions(root='../../COCOData/Data/val2014/',annFile = '../../COCOData/Data/annotations/captions_val2014.json', transform=transforms.ToTensor())        
#         self.testAnns = self.testCOCO.coco.getAnnIds()
#         self.loadEvalBatch()        
        
        
#         self.stopFlag = False
#         self.shuffleInds()


#     def shuffleInds(self):
#         random.shuffle(self.trainAnns)

#     def getBatch(self, num_samples=None):        
#         if num_samples is None:
#             num_samples = self.batchSize

#         if self.epoch > self.maxEpochs:
#             print 'Maximum Epoch Limit reached'
#             self.stopFlag = True
#             return None, None, None, None
        
#         if self.iterInd + num_samples - 1 > self.trainSamples:
#             subsample = self.trainAnns[self.iterInd:]
#         else:
#             subsample = self.trainAnns[self.iterInd:self.iterInd + num_samples]

#         maxLen = 0
#         self.unkCount = 0
#         imgs = []
#         intCaptions = []
#         intTargets = []
#         num_captions = []
        
#         for sample in  self.trainCOCO.coco.loadAnns(subsample):
#         #for sample_id, annot in subsample:                        

#             im_path = os.path.join(self.trainCOCO.root, self.trainCOCO.coco.imgs[sample['image_id']]['file_name'])            
#             if not os.path.isfile(im_path):
#                 continue
#             img = Image.open(im_path).convert('RGB')
#             if self.transform is not None:
#                 img = self.transform(img)
#                 img = img.unsqueeze(0)

            
#             caption = word_tokenize(sample['caption'])                
#             _intCaption = []
#             for word in caption:
#                 try:
#                     _intCaption.append(self.vocab[word.lower().strip()])
#                 except KeyError:
#                     _intCaption.append(self.vocab['<unk>'])
#                     self.unkCount += 1
            
#             intCaption = [self.vocab['<go>']] + _intCaption
#             intTarget = _intCaption + [self.vocab['<stop>']]
#             if maxLen < len(intCaption):
#                 maxLen = len(intCaption)

#             intCaptions.append(intCaption)
#             intTargets.append(intTarget)
                            
#             # num_captions.append(len(annot))
#             imgs.append(img)
            
#         padCaptions = [torch.LongTensor(item + [0] * (maxLen - len(item))) for item in intCaptions]        
#         padTargets = [torch.LongTensor(item + [0] * (maxLen - len(item))) for item in intTargets]
#         batchCaptions = torch.stack(padCaptions)
#         batchTargets = torch.stack(padTargets)
#         batchImgs = torch.cat(imgs, 0)

#         self.globalInd += 1
#         self.iterInd += num_samples
#         if self.iterInd > self.trainSamples:
#             self.iterInd = 0
#             self.shuffleInds()
#             self.epoch += 1
#             self.globalInd = 1
#         else:
#             pass
            
#         #assert batchImgs.size(0) == batchCaptions.size(0) == batchTargets.size(0)
#         return batchImgs, batchCaptions, batchTargets, num_captions

#     def loadEvalBatch(self):
#         random.shuffle(self.testAnns)
#         #select 3 samples at random
#         self.testAnns = self.testAnns[:3]
#         self.testCaptions = []        
#         imgs = []        
#         num_captions = []
#         self.testImID = []
#         for sample in self.testCOCO.coco.loadAnns(self.testAnns):
#             im_path = self.testCOCO.coco.imgs[sample['image_id']]['file_name']
#             self.testImID.append(im_path)
#             img = Image.open(os.path.join(self.testCOCO.root, im_path)).convert('RGB')
#             if self.transform is not None:
#                 img = self.transform(img)

#             imgs.append(img)
#             self.testCaptions.append(sample['caption'])
        
#         self.testImgs = torch.stack(imgs, 0)
#         self.testSeqLens = num_captions

#     # Get a single image-caption pair
#     def getSample(self):        
#         if self.epoch > self.maxEpochs:
#             print 'Maximum Epoch Limit reached'
#             self.stopFlag = True
#             return
        
#         # maxLen = 0
#         self.unkCount = 0
#         imgs = []
#         intCaptions = []
#         intTargets = []
#         # num_captions = []
        
#         sample_id = self.singletrainAnns[self.iterInd:self.iterInd + 1][0]
#         sample = self.trainCOCO.coco.loadAnns(sample_id)[0]
#         im_path = os.path.join(self.trainCOCO.root, self.trainCOCO.coco.imgs[sample['image_id']]['file_name'])            
#         if not os.path.isfile(im_path):
#             while not os.path.isfile(im_path):
#                 self.iterInd += 1
#                 sample = self.singletrainAnns[self.iterInd:self.iterInd + 1]
#                 im_path = os.path.join(self.trainCOCO.root, self.trainCOCO.coco.imgs[sample['image_id']]['file_name'])            

#         img = Image.open(im_path).convert('RGB')
#         if self.transform is not None:
#             img = self.transform(img)
#             img = img.unsqueeze(0)
    
#         caption = word_tokenize(sample['caption'])
#         _intCaption = []
#         for word in caption:
#             try:
#                 _intCaption.append(self.vocab[word.lower().strip()])
#             except KeyError:
#                 _intCaption.append(self.vocab['<unk>'])
#                 self.unkCount += 1
        
#         intCaption = [self.vocab['<go>']] + _intCaption
#         intTarget = _intCaption + [self.vocab['<stop>']]
        
#         intCaptions.append(intCaption)
#         intTargets.append(intTarget)
#         imgs.append(img)
            
#         # padCaptions = [torch.LongTensor(item + [0] * (maxLen - len(item))) for item in intCaptions]

#         #NOTE - Padding targets with arbitrarily large number instead of zero for ease of computing loss
#         # padTargets = [torch.LongTensor(item + [10000] * (maxLen - len(item))) for item in intTargets]
#         batchCaptions = torch.LongTensor(intCaptions)
#         batchTargets = torch.LongTensor(intTargets)
#         batchImgs = torch.cat(imgs, 0)

#         self.globalInd += 1
#         if self.iterInd > self.singletrainSamples:
#             self.iterInd = 0
#             self.shuffleInds()
#             self.epoch += 1
#             self.globalInd = 0
#         else:
#             self.iterInd += 1
#             if self.iterInd > self.singletrainSamples:
#                 self.iterInd = 0
#                 self.shuffleInds()
#                 self.epoch += 1
#                 self.globalInd = 0        
#         return batchImgs, batchCaptions, batchTargets

