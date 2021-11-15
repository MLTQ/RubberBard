from midi_utils import *
import torch
from bard import RubberBardLSTMFC
from bard import RubberBardLSTMFC2
from bard import RubberBardFCFCFC
import glob
import torch.nn.functional as F


path = glob.glob("./pt_files/*.pt")

print(path[0])
# folder = 'zelda_original'
# folder = 'megaman'
# folder = 'gerudo'
#folder = 'DK'
folder = 'zelda'

# Generate total vocab.
tkMsgsList, numUniques, tokenToMsg, msgToToken = generateInputFromSongs(folder)

############
#Ensure these are the same as was used to generate the weights file!
k = 20
CONTEXT_SIZE = k
EMBEDDING_DIM = 10
vocab = numUniques
num_layers = 3
dropout = 0.3
batch_size = k
#################

song_len = 1000
title = 'Prediction_song' + folder

seed_notes = tkMsgsList[0][:batch_size]
seed = torch.tensor(seed_notes)

model = RubberBardLSTMFC2(vocab, EMBEDDING_DIM, CONTEXT_SIZE, num_layers, dropout, batch_size=k)
# model = RubberBardFCFCFC(vocab, EMBEDDING_DIM, CONTEXT_SIZE, num_layers, dropout, batch_size)
# model = RubberBardLSTMFC(vocab, EMBEDDING_DIM, CONTEXT_SIZE, num_layers, dropout, batch_size=k)
# model.load_state_dict(torch.load(path[0]), strict=False)?
model.load_state_dict(torch.load(path[0]), strict=False)

model.eval()
pred = []

#Here we essentially go through the training process on the seed notes, but do not backprop. this generates predictions without altering the model.
for i in range(song_len):
    # print(seed)

    out = model(seed)
    log_prob = F.log_softmax(out.view([1, numUniques]), dim=1)
    values, indices = log_prob[0].max(0)

    new_pred = indices.item()

    print("Predicting......", new_pred)
    pred.append(new_pred)
    seed = torch.cat((seed[1:], torch.tensor([new_pred])))

make_midi(pred,tokenToMsg, title)