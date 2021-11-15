import mido
import glob
from collections import namedtuple

Msg = namedtuple('Msg', 'note dt velocity')

def createMsg(msg):
    if msg.velocity != 0:
        msg.velocity = 127
    return Msg(msg.note, msg.time, msg.velocity)

def tokenize(list_, map):
    return [map[i] for i in list_]


def generateInputFromSongs(folder):
    """
    generate a list of tokenized midi messages from songs in ./songs directory
    a midi message is converted to a custom namedtuple Msg(note, dt, velocity).
    the velocity is reduced to on/off (0 or 127).

    returns:
    tokenizedMsgs - tokenized list of custom midi messages generated from songs in songs file (list)
    numUniques    - number of uniques in tokenizedMsgs (int)
    tokenToMsg    - a dict to convert from tokens to messages (dict)
    msgToToken    - a dict to convert from messages to tokens (dict)
    """
    msgs = []
    uniques = []

    for filename in glob.glob("./songs/"+folder+"/*.mid"):
        # print(filename)
        for msg in mido.MidiFile(filename):
            # print(msg)
            if not msg.is_meta:
                if msg.type == 'note_on':
                    msg_ = createMsg(msg)
                    msgs.append(msg_)
                    if msg_ not in uniques:
                        uniques.append(msg_)

    tokenToMsg = dict(enumerate(uniques))
    msgToToken = {v: k for k, v in tokenToMsg.items()}

    # tokenizedUniques = tokenize(uniques, msgToToken)
    tokenizedMsgs = tokenize(msgs, msgToToken)

    # I return len(uniques) instead of tokenizedUniques because it's just a list [0 .. len(uniques) - 1]
    return tokenizedMsgs, len(uniques), tokenToMsg, msgToToken

# you can use tkMsgs as input.
# you can also perform a sliding window if you want (function not written yet)
#tkMsgs, numUniques, tokenToMsg, msgToToken = generateInputFromSongs()


