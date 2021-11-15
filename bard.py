import torch
import torch.nn as nn
import torch.nn.functional as F

class RubberBardFCFCLSTMFC(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim = 100, num_layers = 3, dropout = 0.3, batch_size = 100):
        super(RubberBardFCFCLSTMFC, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.num_layers = num_layers
        self.dropout = dropout
        self. batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
        self.lstm = nn.LSTM(self.vocab_size, self.hidden_dim, self.num_layers, dropout=self.dropout)
        # Define the output layer
        self.linear3 = nn.Linear(self.hidden_dim, vocab_size)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        out, self.hidden = self.lstm(out.view(-1, 1, out.shape[1]))
        out = self.linear3(out)
        log_probs = F.log_softmax(out[0], dim=1)
      #  log_probs = F.log_softmax(out, dim=1)

        return log_probs, out

class RubberBardLSTMFC(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size, num_layers = 3, dropout = 0.3, batch_size = 100):
        super(RubberBardLSTMFC, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.num_layers = num_layers
        self.dropout = dropout
        self. batch_size = batch_size
        self.hidden_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, vocab_size)
        #self.linear2 = nn.Linear((vocab_size), vocab_size)
        self.lstm = nn.LSTM(self.vocab_size, self.hidden_dim, self.num_layers, dropout=self.dropout)
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                #Setting discrete forget params
                bias.data[start:end].fill_(0.1)
        # Define the output layer
        self.linear3 = nn.Linear(self.hidden_dim, vocab_size)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda(),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda())

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        #out = self.linear2(out)
        out, _ = self.lstm(out.view(-1, 1, out.shape[1]))
        out = self.linear3(out)
        log_probs = F.log_softmax(out[0], dim=1)


        return log_probs, out

class RubberBardFCFCFC(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim = 100, num_layers = 3, dropout = 0.3, batch_size = 100):
        super(RubberBardFCFCFC, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.num_layers = num_layers
        self.dropout = dropout
        self. batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
        # Define the output layer
        self.linear3 = nn.Linear(vocab_size,vocab_size)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        out = self.linear3(out)
        log_probs = F.log_softmax(out, dim=1)
      #  log_probs = F.log_softmax(out, dim=1)

        return log_probs, out


class RubberBardLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size, num_layers = 3, dropout = 0.3, batch_size = 100):
        super(RubberBardLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.num_layers = num_layers
        self.dropout = dropout
        self. batch_size = batch_size
        self.hidden_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        #self.linear1 = nn.Linear(context_size * embedding_dim, vocab_size)
        #self.linear2 = nn.Linear((vocab_size), vocab_size)
        self.lstm = nn.LSTM(self.vocab_size, self.hidden_dim, self.num_layers, dropout=self.dropout)
        # for names in self.lstm._all_weights:
        #     for name in filter(lambda n: "bias" in n, names):
        #         bias = getattr(self.lstm, name)
        #         n = bias.size(0)
        #         start, end = n // 4, n // 2
        #         #Setting discrete forget params
        #         bias.data[start:end].fill_(0.1)
        # Define the output layer
        #self.linear3 = nn.Linear(self.hidden_dim, vocab_size)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        #out = self.linear2(out)
        out, _ = self.lstm(embeds.view(-1, 1, embeds.shape[1]))
        #out = self.linear3(out)
        log_probs = F.log_softmax(out[0], dim=1)


        return log_probs, out



class RubberBardLSTMFC2(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size, num_layers = 3, dropout = 0.3, batch_size = 100):
        super(RubberBardLSTMFC2, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.num_layers = num_layers
        self.dropout = dropout
        self. batch_size = batch_size
        self.hidden_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, vocab_size)
        #self.linear2 = nn.Linear((vocab_size), vocab_size)
        self.lstm = nn.LSTM(self.vocab_size, self.hidden_dim, self.num_layers, dropout=self.dropout)
        # for names in self.lstm._all_weights:
        #     for name in filter(lambda n: "bias" in n, names):
        #         bias = getattr(self.lstm, name)
        #         n = bias.size(0)
        #         start, end = n // 4, n // 2
        #         #Setting discrete forget params
        #         bias.data[start:end].fill_(0.5)
        # # Define the output layer
        self.linear3 = nn.Linear(self.hidden_dim, vocab_size)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda(),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda())

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        #out = self.linear2(out)
        out, _ = self.lstm(out.view(-1, 1, out.shape[1]))
        out = self.linear3(out)

        return out