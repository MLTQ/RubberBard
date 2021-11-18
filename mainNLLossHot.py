import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from bard import *
from tqdm import tqdm
from midi_utils import *
from time import sleep
import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='0'
#My 660 is too old to run CUDA, but my 970 works well!
#Just some house keeping before we start
print('Python VERSION:', sys.version)
print('pyTorch VERSION:', torch.__version__)
print(' Is CUDA available? ')
print(torch.cuda.is_available())
#TODO: Make this actually report version
#print('CUDA VERSION: 10.1 V10.1.105')
from subprocess import call
# Run this command in the python console, I would eval() it, but I won't.
#! nvcc --version
print('CUDNN VERSION:', torch.backends.cudnn.version())
print('Number CUDA Devices:', torch.cuda.device_count())
#print('Devices')
#call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
print('Active CUDA Device: GPU', torch.cuda.current_device())
print ('Available device count', torch.cuda.device_count())
print ('Current cuda device ID', torch.cuda.current_device())
torch.set_default_tensor_type('torch.cuda.FloatTensor')

device = torch.cuda.device(0)
print(torch.cuda.get_device_name(0))




folder = "zelda"
song_length = 10000 #Gerudo Valley is like 10k messages

tkMsgsList, numUniques, tokenToMsg, msgToToken = generateInputFromSongs(folder)
epochs = 300

print("Parsed files with...Uniques: ", numUniques, " and Total Messages: ",len(tkMsgsList[0]))
tkMsgs = tkMsgsList[0]
#make_midi(tkMsgs,tokenToMsg,title)y

#The fundamental idea behind this model is to have a machine predict the next note in a series of notes, specifically,
#it is trying to predict a new note from a given context. The idea is that music is patternful, so given a set of notes
#it should be
botches=[]
k = 20#7 for that big Zelda run
for i in range(len(tkMsgs)-(k+1)):
    yy = tuple([(tkMsgs[i:i+k]), tkMsgs[i+k]])
    botches.append(yy)

vocab = set(tkMsgs)
noteToIDX = {note: i for i, note in enumerate(vocab)}

CONTEXT_SIZE = k
EMBEDDING_DIM = 10

losses = []
loss_function = nn.NLLLoss()
learning_rate = 0.0003
output_dim = numUniques
num_layers = 3
hidden_dim = 512
dropout = 0.3
title = "Practice21_Batch"+str(k)+"_"+str(dropout)+"dropLSTMFC_"+folder

#vocab_size, embedding_dim, context_size, num_layers = 3, dropout = 0.3, batch_size = 100):
model = RubberBardLSTMFC2(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE, num_layers, dropout, batch_size=k)


#model = RubberBardFCFCFC(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE, num_layers, dropout, batch_size=k)
#optimizer = optim.SGD(model.parameters(), lr=learning_rate) #Use for FC models lr = 0.0001
optimizer = optim.Adam(model.parameters(), lr=learning_rate) #Use for LSTM!!!!  lr = 0.01
criterion = nn.NLLLoss()

pred = []
model.load_state_dict(torch.load("pt_files/"+"Practice21_Batch20-3Layer0p3.0dropLSTMFC_zelda_Epoch5.pt"))
#model.load_state_dict(state['state_dict'])
#optimizer.load_state_dict(state['optimizer'])

prevEpochs = 0
for epoch in range(epochs):
    total_loss = 0
    print("\n Training Epoch: ", epoch)
    for context, target in tqdm(botches):
        context_idxs = torch.tensor([noteToIDX[w] for w in context], dtype=torch.int64)
        # context_idxs = context_idxs.long()
        targetOneHot = np.zeros(numUniques)
        targetOneHot[target] = 1
        torch.tensor(targetOneHot)
        model.zero_grad()

        out = model(context_idxs)
        log_prob = F.log_softmax(out.view([1, numUniques]), dim=1)
        values, indices = log_prob[0].max(0)

        new_pred = indices.item()

        pred.append(new_pred)

        loss = criterion(log_prob, torch.tensor(target).view([1]))
       # print("Predicting......", new_pred, ".... With loss....", loss.item())
        criterion(log_prob, torch.tensor(target).view([1]))
        loss.backward()
        optimizer.step()

    losses.append(loss)
    titleE =title+"_Epoch"+str(epoch+prevEpochs)
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()

    }
    torch.save(state, "pt_files/"+titleE+"State"+'.pt')
    torch.save(model.state_dict(),"pt_files/"+titleE+'.pt')
    torch.save(model,"pt_files/"+titleE+"Model"+'.pt' )
    j = 0
    while j < (len(pred) - 1):
        note = pred[j]
        if note == pred[j + 1]:
            del pred[j]
        else:
            j += 1
    make_midi(pred, tokenToMsg, "songs/" + folder + "Gen/" +titleE+".mid")

    # log = open(title+".txt", "a+")
    # log.write(losses)
    # log.write("\n"")
    # pred = []
    print(losses)  # The loss decreased every iteration over the training data!