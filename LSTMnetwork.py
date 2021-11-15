import torch
import torch.nn  as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
torch.manual_seed(1)
#Vocab Size : Num unique notes
#Target Size:1!

'''Assumptions about music:
    We did not find any "note off" signals in our attempts to tokenize MIDI. There were several packages to choose from, 
    most examples of music generation we found used a base tokenization from a library "music21". This would tokenize 
    MIDI signals in to motes, chords, and rests. For our LSTM network topology, we found this was actually increasing 
    the required complexity unnecessarily. Since a chord is just a series of overlapping notes, it makes no sense to 
    encode the chord object and the consituent notes- instead, what we've done is use the "mido" library, which is more
    true to form of how MIDI signals are sent. MIDI doesn't send a message saying "Play this note for this long" like 
    music21 tokenization assumes. MIDI sends a "note on" signal, with a velocity- how "hard" you're hitting the note. 
    Later on, it will send a "note off" message to turn that note off- to wit, you can plug a MIDI keyboard in to a 
    synthesizer, hold a key down, and unplug the keyboard and the synthesizer will continue to play the note until you
    plug the controller back in and hit the note again, sending a quick "note on, note off" series of messages. 
    In our tokenization, we found no "note off" messages, but we did see patterns of "Note X on, velocity 127", followed 
    a few messages later by "Note X on, velocity 0". Since velocity is the intensity of the note played, a velocity of 0
    would correspond to the note turning off! Because this occurs with such regularity, we believe it uneccesary to 
    actually encode duration of note strikes as music21 does, and we will leverage the pattern recognition of the LSTM
    to observe that notes are turned off quickly after they're turned on. 
'''
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim, num_layers=2, embed_size=2, vocab_size=1):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = 0.3
        self.embed_size = embed_size

        self.embed = nn.Embedding(vocab_size, embed_size)
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=self.dropout)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.batch_size, input.shape[1]))#input)
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out,self.hidden= self.lstm(embedded)#input.view(-1, self.batch_size, input.shape[1]))

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out)#lstm_out[-1].view(self.batch_size, -1))
        return y_pred#.view(-1)
####################################################
    # def __init__(self, in_size):
    #     super(ExampleDenseNetwork, self).__init__()
    #     self.fc1 = nn.Linear(in_size, 20)
    #     self.fc2 = nn.Linear(20, 30)
    #     self.fc3 = nn.Linear(30, 4)
    #     self.relu = nn.ReLU()
    #     self.softmax = nn.Softmax(dim=0)
    #     # nn.Linear(input_size,output_size)
    #
    # def forward(self, x):
    #     out = self.fc1(x)
    #     out = self.relu(out)
    #     out = self.fc2(out)
    #     out = self.relu(out)
    #     out = self.fc3(out)
    #     return out
    ###################################
    # def __init__(self, input_size, hidden_size, num_classes, n_layers=2):
    #     super(RNN, self).__init__()
    #
    #     self.input_size = input_size
    #     self.hidden_size = hidden_size
    #     self.num_classes = num_classes
    #     self.n_layers = n_layers
    #
    #     self.notes_encoder = nn.Linear(in_features=input_size, out_features=hidden_size)
    #
    #     self.bn = nn.BatchNorm1d(hidden_size)
    #
    #     self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers)
    #
    #     self.logits_fc = nn.Linear(hidden_size, num_classes)
    #
    # def forward(self, input_sequences, input_sequences_lengths, hidden=None):
    #     batch_size = input_sequences.shape[1]
    #
    #     notes_encoded = self.notes_encoder(input_sequences)
    #
    #     notes_encoded_rolled = notes_encoded.permute(1, 2, 0).contiguous()
    #     notes_encoded_norm = self.bn(notes_encoded_rolled)
    #
    #     notes_encoded_norm_drop = nn.Dropout(0.25)(notes_encoded_norm)
    #     notes_encoded_complete = notes_encoded_norm_drop.permute(2, 0, 1)
    #
    #     # Here we run rnns only on non-padded regions of the batch
    #     packed = torch.nn.utils.rnn.pack_padded_sequence(notes_encoded_complete, input_sequences_lengths)
    #     outputs, hidden = self.lstm(packed, hidden)
    #
    #     # Here we unpack sequence(back to padded)
    #     outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
    #
    #     outputs_norm = self.bn(outputs.permute(1, 2, 0).contiguous())
    #     outputs_drop = nn.Dropout(0.1)(outputs_norm)
    #     logits = self.logits_fc(outputs_drop.permute(2, 0, 1))
    #     logits = logits.transpose(0, 1).contiguous()
    #
    #     neg_logits = (1 - logits)
    #
    #     # Since the BCE loss doesn't support masking,crossentropy is used
    #     binary_logits = torch.stack((logits, neg_logits), dim=3).contiguous()
    #     logits_flatten = binary_logits.view(-1, 2)
    #     return logits_flatten, hidden
    ##############################################
    # def __init__(self, input_dim, hidden_dim, batch_size, output_dim, num_layers):  #vocab_size,  dropProb=0.5, seizure = 0.1):
    #     super(Protobard, self).__init__()
    #
    #
    #     self.input_dim = input_dim
    #     self.hidden_dim = hidden_dim
    #     self.batch_size = batch_size
    #     self.num_layers = num_layers
    #
    #     #Structure: LSTM:FC:Out
    #     self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
    #     #Then a FC, drop out, and probabilistic?
    #     # Idea: Use word embeddings to store musical semantic information? ->Stretch goal
    #     # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
    #
    #     self.linear = nn.Linear(self.hidden_dim, output_dim)
    #
    # def init_hidden(self):
    #     return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
    #             torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
    #
    #
    #
    # def forward(self, input):
    #     #shape: input x batch x hidden
    #     #shape of hidden:(a, b), where a and b both
    #     # have shape (num_layers, batch_size, hidden_dim).
    #     try:
    #         dim1 =
    #     lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
    #
    #     #Only taking the output from the final timestep- maybe take more?
    #     #can pass the entire lstm_out to the next layer if it is a seq2seq prediction
    #     y_pred = self.linear(lstm_out[-1].view(self.batch_size,-1))
    #     return y_pred(-1)
    #
    #



    #
    # def forward(self, sentence):
    #     embeds = self.word_embeddings(sentence)
    #     lstm_out, _ = self.lstm(embeds.view(len(sentence),1,-1))
    #     tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
    #     tag_scores = F.log_softmax(tag_space, dim=1)
    #     return tag_scores
    #
