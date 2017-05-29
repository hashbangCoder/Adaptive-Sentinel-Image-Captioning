import torch
import time
from torch.autograd import Variable
import cPickle as pickle
import  argparse
import math, pdb, os, json
import dataloader
import numpy as np
import models
from torch.nn.utils import clip_grad_norm
from torchvision.utils import save_image
import visdom
parser = argparse.ArgumentParser(description="less script")

parser.add_argument("--num-epochs", dest="epochs", help="Number of epochs", default=15, type=int)
parser.add_argument("-b", "--batch-size", dest="batchSize", help="Mini-batch size", default=40, type=int)
parser.add_argument("--input-size", dest="inputSize", help="Size of input to model", default=300, type=int)
parser.add_argument("--hidden-size", dest="hiddenSize", help="Size of hidden to model", default=512, type=int)
parser.add_argument("--eval-freq", dest="eval_freq", help="How frequently (every mini-batch) to evaluate model", default=400, type=int)

parser.add_argument("--learning-rate", dest="lr", help="Learning Rate for RNN", default=4e-4, type=float)
parser.add_argument("--cnn-learning-rate", dest="cnn_lr", help="Learning Rate for CNN", default=0.00001, type=float)
parser.add_argument("--grad-clip", dest="grad_clip", help="Clip gradients of RNN model", default=0.5, type=float)
parser.add_argument("--dropout", dest="dropout", help="Dropout Probability in RNN + Attention model", default=0.5, type=float)

parser.add_argument("--gpu", dest="gpu", help="GPU device select", default=1)
parser.add_argument("--bootstrap", dest="bootstrap", help="Bootstrap word embeds with GloVe?", default=0, type=int)
parser.add_argument("--use-full-vocab", dest="use_full_vocab", help="use the fulls set of word tokens?", default=0, type=int)
parser.add_argument("--sentinel", dest="sentinel", help="SentinelLSTM or LSTM", default=1, type=int)
parser.add_argument("--layer-norm", dest="layer_norm", help="Layer normalization", default=0, type=int)

parser.add_argument("--load-model", dest="load_model", help="Load saved model", default=None, type=str)
parser.add_argument("--save-dir", dest="save_dir", help="Directory to save trained models", default='Saved-Models', type=str)
parser.add_argument("--title", dest="title", help="Title for visdom figure", default='<>', type=str)
opt = parser.parse_args()

viz = visdom.Visdom()

if opt.use_full_vocab:
    print 'Using full vocabulary...'
    assert os.path.isfile('vocab.pkl') and os.path.isfile('reverse_vocab.pkl')
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('reverse_vocab.pkl', 'rb') as f:
        rVocab = pickle.load(f)
    rVocab[0] = ''
else:
    # Limit vocabulary to words that occur 5 times -- official implementation
    print 'Limitng vocabulary to frequent tokens only...'
    vocab, rVocab = {}, {}
    with open('cocotalk_vocab.json', 'r') as f:
        coco_vocab = json.load(f)
        coco_vocab = coco_vocab['ix_to_word']
        coco_vocab = coco_vocab.values() + ['<go>'] + ['<stop>'] + ['<unk>']

    for ind, v in enumerate(coco_vocab):
        rVocab[ind + 1] = v
        vocab[v] = ind + 1
    rVocab[0] = ''

start_ind = vocab['<go>']
stop_ind = vocab['<stop>']

dl = dataloader.dataloaderBundled(opt.batchSize, opt.epochs, vocab)
wordEmbed = torch.nn.Embedding(len(vocab) + 1, 300, 0)

if opt.bootstrap and not opt.use_full_vocab:
    print 'Bootstrapping with pretrained GloVe word vectors...'
    assert os.path.isfile('embeds_frequent.pkl'), 'Cannot find pretrained Word embeddings to bootstrap'
    with open('embeds_frequent.pkl', 'rb') as f:
        embeds = pickle.load(f)
    assert wordEmbed.weight.size() == embeds.size()
    wordEmbed.weight.data = embeds


crit = torch.nn.CrossEntropyLoss().cuda()

if opt.load_model:
    print 'Loading previously trained models...'
    assert os.path.isfile(opt.load_model), 'Model Path is not valid.'
    net, optimizer = torch.load(opt.load_model)
    cnn = models.buildCNN()
    cnn = cnn.cuda()
else:
    print 'Building Models...'
    cnn = models.buildCNN()
    cnn = cnn.cuda()    
    if opt.sentinel:
        print 'Using the SentinelLSTM-Attention Model'
        opt.save_dir = 'SentAtt-Saved-Models/'
        out_path = 'SentAtt-Output/'
        net = models.SentinelNet(wordEmbed, opt.batchSize, len(vocab), crit, p=opt.dropout, start_ind=start_ind, stop_ind=stop_ind)
    else:
        print 'Using the LSTM-Attention Model'
        out_path = 'RegAtt-Output/'
        opt.save_dir = 'RegAtt-Saved-Models/'
        net = models.regularNet(wordEmbed, opt.batchSize, len(vocab), crit, p=opt.dropout, start_ind=start_ind, stop_ind=stop_ind, layer_norm=opt.layer_norm)
    net = net.cuda()
    # Official Implementation hyperparams
    # optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(0.8, 0.999))


## TODO - 

# def beamSearchEval(net):
#   print 'Model evaluation on mini-test set using beam-search...'
#   net.eval()
#   net.test = True
#   im = Variable(dl.testImgs.cuda(),volatile=True)
#   target_captions = dl.testCaptions
#   start_token = dl.vocab['<go>']
#   stop_token = dl.vocab['<stop>']
#   maxSeqLen = 12
#   for i in range(captions.size(0)):
#       stopFlag = False
#       generated = []
#       caption = Variable(torch.LongTensor([[start_token]]))
#       while maxSeqLen > 0:
#           inds = net(im[i].unsqueeze(0),captions,None,None)
#           list_ind = list(inds)
#           generated.append([rVocab[item%len(vocab)] if item!=stop_token else '' for item in list_ind])
#           captions = Variable(inds.view(net.beamWidth,-1).cuda(),volatile=True)

#           #Break when all candidates output <stop>
#           if len(set(generated)) <= 1:
#               break
#       print '[%d] imageID: %s\t\tGround Truth Caption: %s' %(i+1,dl.testImID[i],target_captions[i])
#       printBeams(generated)
#   net.test = False
#   net.train(True)
#   return

def save_model(net, optimizer, name=None):
    save_dict = dict({'model': net.state_dict(), 'optim': optimizer.state_dict(), 'epoch': dl.epoch})
    print '-' * 60
    print 'Saving Model to : ', opt.save_dir
    if name is not None:
        torch.save(save_dict, opt.save_dir + name)
    else:
        torch.save(save_dict, opt.save_dir + 'saved_model_%d_%d.pth' % (dl.epoch, dl.iterInd))
    print '-' * 60 


def lr_anneal(optim, decay=0.7, lr_floor=0.000001):
    # Taken from https://discuss.pytorch.org/t/adaptive-learning-rate/320/2
    for param_group in optim.param_groups:
        param_group['lr'] = max(lr_floor, decay * param_group['lr'])


def unfreezeCNN(cnn):
    print 'Unfreezing CNN parameters to finetune...'
    for param in cnn.parameters():
        param.requires_grad = True
    return cnn


# Greedy/Stochastic decoder
# Sample = 1,2 [1: Greedy decoder, 2 : Stochastic decoder]
def greedyEval(cnn, net, testIm=None, testCap=None, sample=2, save=False):
    print '\n', '*'*70
    if sample == 1:
        print 'Model evaluation on mini-test set using greedy-decoding...'
    elif sample == 2:
        print 'Model evaluation on mini-test set using Stochastic-decoding...'
    print '\n'
    net.eval()
    cnn.eval()  
    assert net.training is False
    if testIm is not None and testCap is not None:
        im = testIm.cuda()
        target_captions = [[' '.join([rVocab[item] for item in item1])] for item1 in testCap]
        image_ids = ['N.A.'] * len(target_captions)
    else:
        im = dl.testImgs.cuda()
        target_captions = dl.testCaptions
        image_ids = dl.testImID
    
    batchIm = Variable(im.cuda(), volatile=True)    
    for i in range(len(target_captions)):               
        cnnFeats, fc_out = cnn(batchIm[i].unsqueeze(0))
        _, alphas, betas, generated = net(cnnFeats, fc_out, None, None, sample=sample)        
        gen_string = ' '.join([rVocab[item + 1] for item in generated])
        print '[%d] im-ID: %s  \t\tGround Truth: %s' % (i + 1, image_ids[i], target_captions[i])
        print 'Generated: ', gen_string
        print '\n'    
        if save and isinstance(save, str):
            print 'Saving Evaluation Output'
            save_dict = {'alpha': alphas, 'beta': betas, 'output': gen_string, 'image': batchIm[i].data.cpu()}
            torch.save(save_dict, save + 'ModelOut_' + str(dl.epoch) + '_' + str(i) + '.pth')
    print '*' * 70
    net.train(True)
    cnn.train(True)
    return


cnn_finetune = False
trainLoss = []
evalLoss = []

loss = []
win = None
print '\n'
print '#' * 40, ' Start Training ', '#' * 40

while dl.epoch <= opt.epochs:
    # dload_start = time.time()
    im, captions, targets, num_caps = dl.getBundledBatch(opt.batchSize)
    if im is None or captions is None or targets is None:
        break    

    #im, captions, targets = dl.getSample()
    im = Variable(im.cuda())
    captions = Variable(captions.cuda())
    targets = Variable(targets.cuda())
    
    # dload_end = time.time()
    cnnFeats, fc_out = cnn(im)
    
    # comp_start = time.time()
    # TODO: Finetune CNN after 10 epochs using another optim
    if opt.sentinel:
        losses, attn, betas = net(cnnFeats, fc_out, captions, targets, num_caps=num_caps, sample=False)
        batch_loss = torch.cat(losses).mean()
    else:
        losses, attn, num_values = net(cnnFeats, fc_out, captions, targets, num_caps=num_caps, sample=False)    
        batch_loss = torch.stack(losses).sum() / num_values

    batch_loss.backward()
    clip_grad_norm(net.parameters(), opt.grad_clip)
    optimizer.step()    
    optimizer.zero_grad()

    # torch.cuda.synchronize()
    # comp_end = time.time()    
    # print 'Timing Info: \n Dataloading : %f\t\t RNN-Att Fwd + Bkwd : %f' % (dload_end - dload_start, comp_end - comp_start)
    if dl.epoch == 10 and dl.iterInd == 0:
        cnn = unfreezeCNN(cnn)     
        cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=opt.cnn_lr)
        cnn_finetune = True

    if cnn_finetune:
        cnn_optimizer.step()
        cnn_optimizer.zero_grad()

    if dl.globalInd % 100 == 0:        
        print 'Epoch:%d/%d | Batch : %d/%d  \t\tTrain-Loss : %f' % (dl.epoch, opt.epochs, dl.globalInd, (dl.trainSamples / dl.batchSize) + 1, batch_loss.data[0])
        loss.append(batch_loss.cpu().data.tolist()[0])
        if win is None:
            win = viz.line(X=np.arange(1, len(loss) + 1) * 100, Y=np.array(loss), opts=dict(xlabel='Batches', ylabel='Train-Loss', title=opt.title), env=out_path.split('-')[0])
        elif isinstance(win, unicode):
            viz.updateTrace(X=np.arange(1, len(loss) + 1) * 100, Y=np.array(loss), win=win, append=False, env=out_path.split('-')[0])
    
    # save after each epoch
    if dl.globalInd == 1 and dl.epoch > 1:
        save_model(net, optimizer, name=None)
        greedyEval(cnn, net, save=out_path)

    if dl.globalInd % opt.eval_freq == 0:
        greedyEval(cnn, net)
    
    if dl.epoch % 4 == 0 and dl.globalInd == 1:  
        print '*' * 80, '-' * 80
        print '\t\t\t\LR ANNEAL\t\t\t', 
        print  '*' * 80, '-' * 80
        lr_anneal(optimizer, decay=0.65)    

#Evaluate at the end and save model
greedyEval(cnn, net, save=out_path + 'eval_end')
save_model(net, optimizer, name='model_train_end.pth')