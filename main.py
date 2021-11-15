import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from bard import RubberBardLSTMFC
from bard import RubberBardFCFCLSTMFC
from bard import RubberBardFCFCFC
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
print('CUDA VERSION: 10.1 V10.1.105')
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
#title = "Practice18_Batch10-3Layer0p3drop05lrLSTM_"+folder
song_length = 10000 #Gerudo Valley is like 10k messages

tkMsgsList, numUniques, tokenToMsg, msgToToken = generateInputFromSongs(folder)
epochs = 30

print("Parsed files with...Uniques: ", numUniques, " and Total Messages: ",len(tkMsgsList[0]))
tkMsgs = tkMsgsList[0]
#make_midi(tkMsgs,tokenToMsg,title)

#The fundamental idea behind this model is to have a machine predict the next note in a series of notes, specifically,
#it is trying to predict a new note from a given context. The idea is that music is patternful, so given a set of notes
#it should be
botches=[]
k = 10
for i in range(len(tkMsgs)-(k+1)):
    yy = tuple([(tkMsgs[i:i+k]), tkMsgs[i+k]])
    botches.append(yy)

vocab = set(tkMsgs)
noteToIDX = {note: i for i, note in enumerate(vocab)}

CONTEXT_SIZE = k
EMBEDDING_DIM = 5

losses = []
loss_function = nn.NLLLoss()
learning_rate = 0.0003
output_dim = numUniques
num_layers = 2
hidden_dim = 512
dropout = 0.5
title = "Practice20_Batch"+str(k)+"-"+str(num_layers)+"Layer0p"+str(dropout*10)+"dropFCFCFC_"+folder

#vocab_size, embedding_dim, context_size, num_layers = 3, dropout = 0.3, batch_size = 100):
#model = RubberBardLSTMFC(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE, num_layers, dropout, batch_size=k)

model = RubberBardFCFCFC(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE, num_layers, dropout, batch_size=k)
#optimizer = optim.SGD(model.parameters(), lr=learning_rate) #Use for FC models lr = 0.0001
optimizer = optim.Adam(model.parameters(), lr=learning_rate) #Use for LSTM!!!!  lr = 0.01
pred = []
#model.load_state_dict(torch.load("Practice17_Batch10-2Layer0p3.0dropLSTM_zelda_Epoch1.pt"))
#model.train()
for epoch in range(epochs):
    total_loss = 0
    print("\n Training Epoch: ", epoch)
    for context, target in tqdm(botches):

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        context_idxs = torch.tensor([noteToIDX[w] for w in context])

        #Clear that grad! Every time!
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_prob, out = model(context_idxs)
        #values, indices = log_prob[0].max(0)
        values, indices = log_prob[0].max(0)
        new_pred = indices.item()

        #sleep(0.5)
        pred.append(new_pred)
        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        loss = loss_function(log_prob, torch.tensor([noteToIDX[target]]))
        print("Predicting......", new_pred, ".... With loss....",loss.item())
        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    losses.append(total_loss)
    titleE =title+"_Epoch"+str(epoch+2)
    torch.save(model.state_dict(),titleE+'.pt')
    # if len(pred)>10000:
    #     #Make a song of the last 10k notes if there are more than 10k
    #     make_midi(pred[-10000:-0], tokenToMsg, titleE)
    # else:
    make_midi(pred, tokenToMsg, titleE)
    pred = []
print(losses)  # The loss decreased every iteration over the training data!














