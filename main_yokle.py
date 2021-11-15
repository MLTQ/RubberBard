from LSTMnetwork import LSTM
import torch
import torch.nn  as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import midoparser as mp
import relu

learning_rate = 0.001
#notes = mp.get_notes("zelda")

tkMsgs, numUniques, tokenToMsg, msgToToken = mp.generateInputFromSongs('zelda')
songNpy = np.zeros(len(tkMsgs))
songNpy += tkMsgs
songTens = torch.from_numpy(songNpy).float()

lstm_input_size = 200
num_train = 3#?
output_dim = numUniques
num_layers = 3
hidden_dim = 88
#word_to_ix = {"hello": 0, "world": 1}
# embeds = nn.Embedding(numUniques, 3)  # N words in vocab, 3 dimensional embeddings
# lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
# hello_embed = embeds(lookup_tensor)


botches=[]
k = 100
for i in range(len(tkMsgs)-101):
    yy = tuple([(tkMsgs[:k]), tkMsgs[k]])
    botches.append(yy)
# bar = [([tkMsgs[i], tkMsgs[i+1]], tkMsgs[i + 2]) for i in range(len(tkMsgs) - 2)]
# foo = [[tkMsgs[i+j] for j in range(lstm_input_size)] for i in range(len(tkMsgs)-lstm_input_size)]
# foo = [tuple(item) for item in foo]
# bar = []
# for i in range(len(foo)):
#     bar.append([foo[i], tkMsgs[i+101]])
# bar = tuple(bar)

vocab = set(tkMsgs)
noteToIDX = {note: i for i, note in enumerate(vocab)}



CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

losses = []
loss_function = nn.NLLLoss()
model = relu.NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)
pred = []
for epoch in range(10):
    total_loss = 0
    for context, target in botches:

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        context_idxs = torch.tensor([noteToIDX[w] for w in context], dtype=torch.long)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_prob, out = model(context_idxs)
        values, indices = log_prob[0].max(0)
        new_pred = indices.item()
        print("Predicting......", new_pred)
        pred.append(new_pred)
        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        loss = loss_function(log_prob, torch.tensor([noteToIDX[target]], dtype=torch.long))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    losses.append(total_loss)
print(losses)  # The loss decreased every iteration over the training data!




















model = LSTM(lstm_input_size, hidden_dim, batch_size=num_train, output_dim=output_dim, num_layers=num_layers, embed_size = 3, vocab_size=numUniques)
#model = Protobard(lstmInputSize, hiddenDim, batch_size=numTrain, output_dim=output_dim, num_layers=num_layers)

loss_fn = torch.nn.MSELoss(size_average=False)

optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

#####################
# Train model
#####################
num_epochs = 2
xx = np.asarray(tkMsgs[0:200])
yy = np.asarray(tkMsgs[1:201])
zz = np.asarray(tkMsgs[2:202])
msgs = np.stack((xx, yy, zz))
X_train = torch.from_numpy(msgs).float()

y_train = np.ndarray([tkMsgs[201], tkMsgs[202], tkMsgs[203]])
hist = np.zeros(num_epochs)

for t in range(num_epochs):
    # Clear stored gradient
    model.zero_grad()

    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    model.hidden = model.init_hidden()

    # Forward pass
    y_pred = model(X_train)

    loss = loss_fn(y_pred, y_train)
    if t % 100 == 0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimiser.step()

x=['lol']