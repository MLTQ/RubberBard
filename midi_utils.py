import mido
import glob
import numpy as np
from collections import namedtuple, deque
from tqdm import tqdm
Msg = namedtuple('Msg', 'type note dt velocity')

def createMsg(msg,tempo, tpb):
    msg_type = 'note_off'
    msg_time = mido.second2tick(msg.time,tpb,tempo)
    if msg.velocity != 0:
        msg.velocity = 64
        msg_type = 'note_on'

    return Msg(msg_type, msg.note, msg_time, msg.velocity)


def tokenize(song, map):
    return [map[msg] for msg in song]


def generateInputFromSongs(folder):
    """
    generate a list of tokenized midi messages from songs in ./songs directory
    a midi message is converted to a custom namedtuple Msg(note, dt, velocity).
    the velocity is reduced to on/off (0 or 127).

    returns:
    tokenizedSongs - tokenized list of lists of custom midi messages generated from songs in songs file (list[list[Msg]])
    numUniques     - number of uniques in tokenizedMsgs (int)
    tokenToMsg     - a dict to convert from tokens to messages (di'pip3' is not recognized as an internal or external command,
operable program or batch file.
ct)
    msgToToken     - a dict to convert from messages to tokens (dict)
    """
    songs = []
    uniques = []
    print("Loading files...")
    for filename in glob.glob("./songs/"+folder+"/*.mid"):

        print(filename)

        song = []
        midi_msg = mido.MidiFile(filename)
        tpb = midi_msg.ticks_per_beat
        tempo = 250000/2

        for msg in midi_msg:

            # print(msg)

            if not msg.is_meta:

                if msg.type == 'note_on' or msg.type == 'note_off':
                    # print(msg.type)
                    msg_ = createMsg(msg,tempo,tpb)
                    # print(msg_)

                    songs.append(msg_)

                    if msg_ not in uniques:
                        uniques.append(msg_)

        # songs.append(song)

    tokenToMsg = dict(enumerate(uniques))
    msgToToken = {v: k for k, v in tokenToMsg.items()}

    # tokenizedUniques = tokenize(uniques, msgToToken)
    tokenizedSongs = []

    # for song in songs:
    tokenizedSongs.append(tokenize(songs, msgToToken))

    # I return len(uniques) instead of tokenizedUniques because it's just a list [0 .. len(uniques) - 1]
    return tokenizedSongs, len(uniques), tokenToMsg, msgToToken


# you can use tkMsgs as input.
# you can also perform a sliding window if you want (function not written yet)

#tkSongs, numUniques, tokenToMsg, msgToToken = generateInputFromSongs()


def window(seq, n=100):

    currIndex = 0
    index = 0
    windows = []
    size = len(seq)

    assert (size > n)
    win = []
    labels = []

    while (currIndex + index < size):

        if index % n == 0 and index != 0:
            currIndex += 1
            index = 0

            if win != []:
                windows.append(win)

            win = []

        val = seq[currIndex + index]
        win.append(val)

        if len(win) == n and currIndex != 0:
            labels.append(val)

        index += 1

    return windows, labels


def formatInput(songs, windowSize):
    """
    takes in a list of list of tokenized songs list[list[int]] and windowSize(int).
    returns:
    songChunks : list[list[int]] - a list of lists(len = windowSize)
    labels     : list[int]       - a list of ints(lables) for each songchunck
    len(songChunks) == len(labels)
    """

    labels = []

    for i, song in enumerate(tkSongs):
        tkSongs[i], label = window(song, 100)
        labels.append(label)

    flatten = lambda l: [item for sublist in l for item in sublist]
    songChunks = flatten(tkSongs)
    labels = flatten(labels)

    return songChunks, labels


def save_notes():

    tokenizedSongs, uniques, tokenToMsg, msgToToken = generateInputFromSongs()
    note_tokens = open('notes.txt', 'w')
    note_tokens.write(str(tokenizedSongs))

    return tokenizedSongs, uniques, tokenToMsg, msgToToken


def make_midi(notes,tokenToMsg,title):

    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    notes = np.asarray(notes)
   # i = 0
    # print(notes.size)
    print("\nSaving ", title)
    #for lst in tqdm(notes):

    for i in tqdm(range(notes.size)):
        integ = notes[i]
        track.append(mido.Message(tokenToMsg[integ][0], note=tokenToMsg[integ][1], velocity=tokenToMsg[integ][3], time=round(tokenToMsg[integ][2])))
        # track.append(Message('note_off', note=int(tokenToMsg[integ][1]), velocity=0, time=64))
        #i +=1

    mid.save(title+'.mid')

