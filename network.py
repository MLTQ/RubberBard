import torch
import torch.nn  as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

lstm = nn.LSTM(3, 3) #in 3 out 3
inputs = [torch.randn(1, 3) for _ in range(5)] #this generates a seq of length 5

#initialize the hidden state
hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))
for i in inputs:
    #step through sequence one at a time
    #after each step, hidden <= hidden state
    out, hidden = lstm(i.view(1, 1, -1), hidden)

inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1,1,3), torch.randn(1,1,3)) # cleans the hidden state?
out, hidden = lstm(inputs, hidden)
print('Out')
print(out)
print('hidden')
print(hidden)

def prepare_sequences( seq, to_idx):
    idxs = [to_idx[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

training_data = [("The dog ate the apple".split(),["DET", "NN", "V","DET","NN"]),
                ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])]
testing_data = ["dog ate everybody apple apple apples this book bog".split()]
word_to_idx = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
for word in testing_data[0]:
    if word not in word_to_idx:
        word_to_idx[word] = len(word_to_idx)

print(word_to_idx)
tag_to_idx = {"DET":0, "NN":1, "V": 2}
EMBEDDING_DIM = 8
HIDDEN_DIM = 8


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence),1,-1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_idx), len(tag_to_idx))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)

with torch.no_grad():
    inputs = prepare_sequences(training_data[0][0], word_to_idx)
    tag_scores = model(inputs)
    print(tag_scores)

for epoch in range(1000):
    for sentence, tags in training_data:
        model.zero_grad()

        sentence_in = prepare_sequences(sentence, word_to_idx)
        targets = prepare_sequences(tags, tag_to_idx)

        tag_scores = model(sentence_in)

        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()


with torch.no_grad():
    inputs = prepare_sequences(training_data[0][0], word_to_idx)
    tag_scores= model(inputs)
    print("This is a tag matrix, the closest to 0 is the most likely guess.")
    print("The horizontal axis is the options for the classifier")
    print("Here, the options are \"Article\", \"Noun\", \"Verb\"")
    print("The vertical axis is the index of the sentence passed")
    print("Sentence is \"The dog ate the apple\" => 1 2 3 4 5")
    print(tag_scores)
    print("Now with my own testing data...")
    print("dog ate everybody apple apple apples this book bog")

    inputs = prepare_sequences( testing_data[0], word_to_idx)
    tag_scores = model(inputs)
    print(tag_scores)