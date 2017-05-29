import torch,pdb, torchvision, math
import torch.nn as nn
from  torch.autograd import Variable
from torch.nn import Parameter
from torch.utils.serialization import load_lua
import torch.nn.functional as F
from torch.nn import init


class Bottle(nn.Module):
    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
        return out.view(size[0], size[1], -1)


class Linear(Bottle, nn.Linear):
    pass


def loadResnet():
    resnet = torchvision.models.resnet152(pretrained=True)
    pool, fc = list(resnet.children())[-2:]
    modifiedNet = nn.Sequential(*list(resnet.children())[:-2])

    # Freeze layers
    for param in modifiedNet.parameters():
        param.requires_grad = False

    for param in fc.parameters():
        param.requires_grad = False
    return modifiedNet, pool, fc


class buildCNN(nn.Module):
    def __init__(self):
        super(buildCNN, self).__init__()
        self.conv, self.pool, self.fc = loadResnet()

    def forward(self, _inp):
        conv_out = self.conv(_inp)        
        pool_out = self.pool(conv_out)
        fc_out = self.fc(pool_out.squeeze(2).squeeze(2))
        
        return conv_out, fc_out


class CNNFeatNet(nn.Module):
    def __init__(self, cnnFeatSize, numCNNFeats, outSize, embedSize, p):
        super(CNNFeatNet, self).__init__()
        self.cnnFeatSize = cnnFeatSize
        self.numCNNFeats = numCNNFeats
                
        self.W_a = Linear(cnnFeatSize, outSize)          # inp size - B*49*2048, out size - B*49*512 needs to become B*512*49
        self.W_b = Linear(cnnFeatSize, outSize)
        # self.fc_proj = Linear(1000, embedSize)             # inp size - B*1000, out size - B*512 

        self.W_a_dropout = nn.Dropout(p)
        self.W_b_dropout = nn.Dropout(p)
        # self.fcDropout = nn.Dropout(p)

        self.relu = torch.nn.ReLU()
        
    def forward(self, imFeats):        
        imFeats = imFeats.view(-1, self.cnnFeatSize, self.numCNNFeats)    # size - B*2048*49
        globalImFeat = self.W_b_dropout(self.relu(self.W_b(imFeats.mean(2).squeeze(2))))
        
        localImFeats = self.W_a_dropout(self.relu(self.W_a(imFeats.transpose(2, 1).contiguous())))
        spatialImFeat = localImFeats.transpose(2, 1)                          # size - B*512*49

        # fc_embed = self.fcDropout(self.fc_proj(fc_feat))
        
        return spatialImFeat, globalImFeat,  #fc_embed


class rnnAttnModel(nn.Module):
    def __init__(self, inputSize, hiddenSize, numCNNFeats, vocabSize, p):
        super(rnnAttnModel, self).__init__()
        # d - hiddenSize ; k - numCNNFeats 
        self.d = hiddenSize
        self.k = numCNNFeats
        self.vocabSize = vocabSize
        self.rnn = sentinelLSTMCell(inputSize, hiddenSize)
        self.W_v = Linear(self.d, self.k)            # inp size - B*512*49, outsize - B*49*49
        
        self.W_g = Linear(self.d, self.k)            # inpsize - B*512*1   outSize- B*k*1        
        self.W_s = Linear(self.d, self.k)           # inpsize - B*512      outsize - B*k
        self.w_h = Linear(self.k, 1)                 # inpsize -B*k*k  outsize- B*k*1 should be  B*1*k

        self.ones = Variable(torch.cuda.FloatTensor(1, 1, self.k), requires_grad=False)
        self.W_p = Linear(self.d, self.vocabSize)
        self.dropout_z = nn.Dropout(p)
        self.dropout_b = nn.Dropout(p)
    
    def forward(self, V, hout, sentinel):
        # hidden : B*d*1        
        z_t1 = self.W_v(V.transpose(2, 1).contiguous())
        z_t2 = self.W_g(hout).unsqueeze(2)
        z_t2_resh = z_t2.bmm(self.ones.expand(z_t2.size(0), 1, self.k))
        z_t = self.w_h(self.dropout_z(F.tanh(z_t1 + z_t2_resh).squeeze(2)))                         # inpsize - B*k*k, outsize -B*k

        beta_1 = self.W_s(sentinel).unsqueeze(2)        
        beta_pre = self.w_h(self.dropout_b(F.tanh(beta_1 + z_t2).squeeze(2)))           # inpsize - B*k   outsize - B*1        
        alpha = F.softmax(torch.cat((z_t, beta_pre.unsqueeze(2)), 1).squeeze(2))     # size : B*(k+1)*1
        alpha = alpha.unsqueeze(2)
        beta = alpha[:, -1, :]                                # size - B*k*1
        alpha = alpha[:, :-1, :]                              # size - B*1*1

        context = V.bmm(alpha)                # B*d*k x B*k*1 --> B*d*1
        adaptive_context = (beta.expand_as(sentinel))*sentinel + (1 - beta.expand_as(sentinel))*context           # size : B*d
        logits = self.W_p(adaptive_context + hout)
        return logits, alpha, beta


class sentinelLSTMCell(nn.Module):
    def __init__(self, inputSize, hiddenSize, bias=None):
        super(sentinelLSTMCell, self).__init__()
        self.hiddenSize = hiddenSize
        self.inputSize = inputSize

        self.w_ih = Parameter(torch.Tensor(5 * self.hiddenSize, self.inputSize))
        self.w_hh = Parameter(torch.Tensor(5 * self.hiddenSize, self.hiddenSize))
        self.w_fch = Parameter(torch.Tensor(5 * self.hiddenSize, self.hiddenSize))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hiddenSize)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    # Follow the official code implementation ; use fc features and x[t] as inputs
    def forward(self, inp, states, fc_feat):
        
        hx, cx = states
        gates = F.linear(inp, self.w_ih) + F.linear(hx, self.w_hh) + F.linear(fc_feat, self.w_fch)
        ingate, forgetgate, cellgate, outgate, sgate = gates.chunk(5, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        sgate = F.sigmoid(sgate)
        
        cy = (forgetgate * cx) + (ingate * cellgate)
        sentinel = sgate * F.tanh(cy)
        hy = outgate * F.tanh(cy)

        return hy, cy, sentinel


class SentinelNet(nn.Module):
    def __init__(self, wordEmbed, batchSize, vocabSize, crit, beamWidth=4, p=0.25, maxLen=18, start_ind=None, stop_ind=None):
        super(SentinelNet, self).__init__()
        self.cnnFeatSize = 2048
        self.numCNNFeats = 49
        self.hiddenSize = 512
        self.embedSize = 300
        self.beamWidth = beamWidth
        self.cnn_fc = 2048
        
        self.rnn = sentinelLSTMCell(self.embedSize, self.hiddenSize)
        # self.CtoHDropout = nn.Dropout(p)
        self.cnnToHiddenLinear = Linear(self.cnn_fc, self.hiddenSize)
        self.featNet = CNNFeatNet(self.cnnFeatSize, self.numCNNFeats, self.hiddenSize, self.embedSize, p)
        self.rnnAttnModel = rnnAttnModel(self.embedSize, self.hiddenSize, self.numCNNFeats, vocabSize, p)
        self.wordEmbed = wordEmbed
        self.crit = crit        

        # max timesteps to sample during inference
        self.maxLen = maxLen
        self.start_ind = start_ind
        self.stop_ind = stop_ind

    def getInitHidden(self, batchSize):
        return torch.cuda.FloatTensor(batchSize, self.hiddenSize).fill_(0), torch.cuda.FloatTensor(batchSize, self.hiddenSize).fill_(0)

    def forward(self, cnnFeats, fc_out, all_captions, all_targets, num_caps=None, lens=None, sample=False):
        all_SpatialImFeat, all_globalImFeat = self.featNet(cnnFeats)    # size - B*512*49 , B*512        
        attn_weights = []
        losses = []
        betas = []
        
        if not sample:
            caption = all_captions
            targets = all_targets
            batchSize = all_captions.size(0)
            max_time_steps = all_captions.size(1)
            SpatialImFeat, globalImFeat = [], []
            
            for i, nc in enumerate(num_caps):                
                SpatialImFeat.append(all_SpatialImFeat[i, :].unsqueeze(0).expand(nc, all_SpatialImFeat.size(1), all_SpatialImFeat.size(2)))
                globalImFeat.append(all_globalImFeat[i, :].unsqueeze(0).expand(nc, all_globalImFeat.size(1)))

            SpatialImFeat = torch.cat(SpatialImFeat, 0)
            globalImFeat = torch.cat(globalImFeat, 0)
            h0, c0 = self.getInitHidden(batchSize)
            h0, c0 = Variable(h0), Variable(c0)
            hidden = (h0, c0)
            embedCaptions = self.wordEmbed(caption)
            
            for ind in range(max_time_steps):
                nonzeroInds = targets[:, ind].data.nonzero().squeeze()
                rnnBatch = embedCaptions[:, ind, :]
                hout, cout, sentinel = self.rnn(rnnBatch, hidden, globalImFeat)
                logits, alpha, beta = self.rnnAttnModel(SpatialImFeat, hout, sentinel)

                hidden = (hout, cout)
                attn_weights.append(alpha.data)
                betas.append(beta.data)
                          
                # Compute loss internally w/o an extra loop                
                losses.append(self.crit(logits[nonzeroInds], targets[:, ind][nonzeroInds] - 1))     # Decrease by 1 for CrossEntropyLoss's indices
                # _, pred_inds = F.softmax(logits[nonzeroInds]).max(1)
            return losses, attn_weights, beta   # pred_inds

        else:
            h0, c0 = self.getInitHidden(1)
            h0, c0 = Variable(h0, volatile=True), Variable(c0, volatile=True)
            hidden = (h0, c0)
            generated = []
            token = Variable(torch.cuda.LongTensor([[self.start_ind]]), volatile=True)                  
            alphas, betas = [], []
            for _ in range(self.maxLen):
                embedCaptions = self.wordEmbed(token)
                rnnBatch = embedCaptions[:, 0, :]                
                hout, cout, sentinel = self.rnn(rnnBatch, hidden, all_globalImFeat)
                logits, alpha, beta = self.rnnAttnModel(all_SpatialImFeat, hout, sentinel)
                alphas.append(alpha.data.cpu())
                betas.append(beta.data.cpu())
# ------------------------------------------------------------------------------------------------
                if sample == 1:
                    _, inds = F.softmax(logits).view(1, -1).max(1)
                elif sample == 2:
                # Multinomial
                    inds = F.softmax(logits).view(1, -1).multinomial(1)                                
                else: 
                    raise Exception('Invalid sample option')
# ------------------------------------------------------------------------------------------------
                hidden = (hout, cout)
                inds = inds.squeeze()
                list_ind = inds.data.tolist()[0]
                # generated.append(rVocab[list_ind])
                generated.append(list_ind)
                if list_ind + 1 == self.stop_ind:
                    break

                # break from graph history for next input
                token = Variable(inds.data.unsqueeze(1), volatile=True)
            
            return hidden, alphas, betas, generated
            pass
                
        
class regularAttnModel(nn.Module):
    def __init__(self, inputSize, hiddenSize, numCNNFeats, vocabSize, p):
        super(regularAttnModel, self).__init__()
        # d - hiddenSize ; k - numCNNFeats 
        self.d = hiddenSize
        self.k = numCNNFeats
        self.vocabSize = vocabSize
        
        self.W_v = Linear(self.d, self.k)            # inp size - B*512*49, outsize - B*49*49
        
        self.W_g = Linear(self.d, self.k)            # inpsize - B*512*1   outSize- B*k*1
        
        self.W_s = Linear(self.d, self.k)           # inpsize - B*512      outsize - B*k
        self.w_h = Linear(self.k, 1)                 # inpsize -B*k*k  outsize- B*k*1 should be  B*1*k

        self.ones = Variable(torch.cuda.FloatTensor(1, 1, self.k), requires_grad=False)
        self.W_p = Linear(self.d, self.vocabSize)
        self.dropout_z = nn.Dropout(p)
        self.dropout_b = nn.Dropout(p)
    
    def forward(self, V, hout):
        # hidden : B*d*1        
        
        z_t1 = self.W_v(V.transpose(2, 1).contiguous())
        z_t2 = self.W_g(hout).unsqueeze(2)
        z_t2_resh = torch.bmm(z_t2, self.ones.expand(z_t2.size(0), 1, self.k))
        z_t = self.w_h(self.dropout_z(F.tanh(z_t1 + z_t2_resh).squeeze(2)))                         # inpsize - B*k*k, outsize -B*k

        alpha = F.softmax(z_t.squeeze(2))     # size : B*(k+1)*1                    
        alpha = alpha.unsqueeze(2)
        context = torch.bmm(V, alpha).squeeze(2)                # B*d*k x B*k*1 --> B*d*1        
        logits = self.W_p(context + hout)
        return logits, alpha


class regularNet(nn.Module):
    def __init__(self, wordEmbed, batchSize, vocabSize, crit, beamWidth=4, p=0.25, maxLen=18, start_ind=None, stop_ind=None, layer_norm=False):
        super(regularNet, self).__init__()
        self.cnnFeatSize = 2048
        self.numCNNFeats = 49
        self.hiddenSize = 512
        self.embedSize = 300
        self.beamWidth = beamWidth
        self.cnn_fc = 2048
        if layer_norm:
            print 'Using Layer Normalization in LSTM...'
            cell = LayerNormLSTMCell(self.embedSize + self.hiddenSize, self.hiddenSize)
            self.rnn = LayerNormLSTM(cell, self.embedSize + self.hiddenSize, self.hiddenSize)
        else:
            self.rnn = nn.LSTM(self.embedSize + self.hiddenSize, self.hiddenSize, batch_first=True)
        self.CtoHDropout = nn.Dropout(p)
        self.cnnToHiddenLinear = Linear(self.cnn_fc, self.hiddenSize)
        self.featNet = CNNFeatNet(self.cnnFeatSize, self.numCNNFeats, self.hiddenSize, self.embedSize, p)
        self.rnnAttnModel = regularAttnModel(self.embedSize, self.hiddenSize, self.numCNNFeats, vocabSize, p)
        self.wordEmbed = wordEmbed
        self.crit = crit

        # max timesteps to sampling during inference
        self.maxLen = maxLen
        self.start_ind = start_ind
        self.stop_ind = stop_ind

    def getInitHidden(self, batchSize):
        return torch.zeros(1, batchSize, self.hiddenSize).cuda(), torch.zeros(1, batchSize, self.hiddenSize) .cuda()

    def forward(self, cnnFeats, fc_out, caption, targets, num_caps=None, sample=False):
        # caption is B*maxlen
        all_V, all_gFeats = self.featNet(cnnFeats)    # size - B*512*49 , B*512        
        if not sample:            
            V, gFeats = [], []
            for i, nc in enumerate(num_caps):
                V.append(all_V[i, :].unsqueeze(0).expand(nc, all_V.size(1), all_V.size(2)))
                gFeats.append(all_gFeats[i, :].unsqueeze(0).expand(nc, all_gFeats.size(1)))

            V = torch.cat(V, 0)
            gFeats = torch.cat(gFeats, 0)
                    
            time_steps, batchSize = caption.size(1), caption.size(0)
            gFeats = gFeats.unsqueeze(1).expand(batchSize, time_steps, self.hiddenSize)
            attn_weights = []
        
            nonzeroInds = caption.detach().gt(0)
            exp_nonzeroInds = nonzeroInds.unsqueeze(2).expand(batchSize, time_steps, self.hiddenSize + self.embedSize)
            embedCaptions = self.wordEmbed(caption)
            concatBatch = torch.cat((embedCaptions, gFeats), 2)
            rnnBatch = concatBatch * exp_nonzeroInds.float()
            
            hout, _ = self.rnn(rnnBatch)
            loss = []
            for ind in range(time_steps):                
                logits, alpha = self.rnnAttnModel(V, hout[:, ind, :])                

                # Manually implement CE loss with log softmax
                logsm_logits = -1 * F.log_softmax(logits)

                attn_weights.append(alpha.data)                
                loss.append((logsm_logits.gather(1, (targets[:, ind].unsqueeze(1) - 1).clamp(0, logsm_logits.size(1) - 1)) * nonzeroInds[:, ind].float()).mean())
            
            return loss, attn_weights, nonzeroInds.float().sum()

        else:
            V = all_V
            gFeats = all_gFeats

            h0, c0 = self.getInitHidden(1)
            h0, c0 = Variable(h0), Variable(c0)
            hidden = (h0, c0)
            generated = []            
            token = Variable(torch.cuda.LongTensor([[self.start_ind]]), volatile=True)      
            gFeats = gFeats.unsqueeze(1)
            
            alphas = []
            for _ in range(self.maxLen):
                embedCaptions = self.wordEmbed(token)
                rnnBatch = torch.cat((embedCaptions, gFeats), 2)
                # rnnBatch = concatBatch[:, 0, :]                
                hout, hidden = self.rnn(rnnBatch, hidden)
                logits, alpha = self.rnnAttnModel(V, hout[:, 0, :])
                alphas.append(alpha.data)
# ------------------------------------------------------------------------------------------------
                # Greedy max
                if sample == 1:
                    _, inds = F.softmax(logits).view(1, -1).max(1)
                # stochastic
                elif sample == 2:                
                    inds = F.softmax(logits).view(1, -1).multinomial(1)                                
                else:
                    raise Exception('Invalid sample option')                    
# ------------------------------------------------------------------------------------------------                    
                inds = inds.squeeze()
                list_ind = inds.data.tolist()[0]
                                
                # add one because of zero-indexing
                generated.append(list_ind)
                if list_ind + 1 == self.stop_ind:
                    break

                # break from graph history for next input
                token = Variable(inds.data.unsqueeze(1), volatile=True)
            
            return hidden, alphas, None, generated
            pass


##########################################################################
# Layer normalized LSTM Implementation
class LayerNormLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, ):
        super(LayerNormLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        # LN params
        self.alpha1 = nn.Parameter(torch.Tensor(1, 4 * hidden_size))
        self.beta1 = nn.Parameter(torch.Tensor(1, 4 * hidden_size))

        self.alpha2 = nn.Parameter(torch.Tensor(1, 4 * hidden_size))
        self.beta2 = nn.Parameter(torch.Tensor(1, 4 * hidden_size))

        self.alpha_out = nn.Parameter(torch.Tensor(1, hidden_size))
        self.beta_out = nn.Parameter(torch.Tensor(1, hidden_size))

        # LSTM params
        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size, 4 * hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(1, 4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(hidden_size, 4 * hidden_size))

        self.init_params()

    def init_params(self):
        self.weight_ih.data.set_(init.xavier_uniform(torch.FloatTensor(*self.weight_ih.size())))
        self.weight_hh.data = (init.xavier_uniform(torch.FloatTensor(self.hidden_size, 4 * self.hidden_size)))
        self.bias.data.fill_(0)

        self.alpha1.data.fill_(0)
        self.beta1.data.fill_(1)
        self.alpha2.data.fill_(0)
        self.beta2.data.fill_(1)
        self.alpha_out.data.fill_(0)
        self.beta_out.data.fill_(1)

    def apply_LN(self, _input, alpha, beta, epsilon=1e-5):
        # Input - B * Hidden
        mean = _input.mean(1)
        var = _input.var(1)
        # pdb.set_trace
        out = (((_input - mean.expand_as(_input)) / (epsilon + var.expand_as(_input))) * alpha.expand_as(_input)) + beta.expand_as(_input)
        return out
    
    def forward(self, _input, hx):
        # input is of size - B*inp_size
        h0, c0 = hx        
        # batch_size = _input.size(0)
        wh = torch.mm(h0, self.weight_hh)
        wi = torch.mm(_input, self.weight_ih)

        # LN         
        gates = self.apply_LN(wh, self.alpha1, self.beta1) + self.apply_LN(wi, self.alpha2, self.beta2) + self.bias.expand_as(wh)
        f, i, o, g = gates.chunk(4, dim=1)
        ct = (F.sigmoid(f) * c0) + (F.sigmoid(i) * F.tanh(g))
        ht = F.sigmoid(o) * F.tanh(self.apply_LN(ct, self.alpha_out, self.beta_out))
        return ht, ct


class LayerNormLSTM(nn.Module):
    def __init__(self, cell, input_size, hidden_size):
        super(LayerNormLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = cell

    def forward(self, _input, hx=None, return_last=False, seq_lens=None):
        batch_size, max_time = _input.size(0), _input.size(1)
        if hx is None and _input.is_cuda:
            hx = (Variable(torch.cuda.FloatTensor(batch_size, self.hidden_size).fill_(0)), Variable((torch.cuda.FloatTensor(batch_size, self.hidden_size).fill_(0))))
        outputs = []
        for t in range(max_time):
            hx = self.cell(_input[:, t, :], hx)
            outputs.append(hx[0])
        
        outputs = torch.stack(outputs, 1)   # size - B * T * H
        if return_last:
            assert seq_lens is not None and len(seq_lens) == batch_size, 'Require sequence lengths to return last hidden state'                    
            seq_lens = seq_lens.view(batch_size, 1, 1).expand(batch_size, 1, self.hidden_size)
            return outputs.gather(1, seq_lens).squeeze(1)

        return outputs, None
