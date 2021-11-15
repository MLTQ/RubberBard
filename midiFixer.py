from midi_utils import *


def fixSongs(folder):
    """
    generate a list of tokenized midi messages from songs in ./songs directory
    a midi message is converted to a custom namedtuple Msg(note, dt, velocity).
    the velocity is reduced to on/off (0 or 127).

    returns:
    tokenizedSongs - tokenized list of lists of custom midi messages generated from songs in songs file (list[list[Msg]])
    numUniques     - number of uniques in tokenizedMsgs (int)
    tokenToMsg     - a dict to convert from tokens to messages (dict)
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
        tempo = 2500000*2

        for msg in midi_msg:

            # print(msg)

            if not msg.is_meta:

                if msg.type == 'note_on' or msg.type == 'note_off':
                    # print(msg.type)
                    msg_ = createMsg(msg,tempo,tpb)
                    # print(msg_)

                    song.append(msg_)

                    if msg_ not in uniques:
                        uniques.append(msg_)




        # songs.append(song)

        tokenToMsg = dict(enumerate(uniques))
        msgToToken = {v: k for k, v in tokenToMsg.items()}

    # tokenizedUniques = tokenize(uniques, msgToToken)
        tokenizedSongs = []

    # for song in songs:
        tokenizedSongs.append(tokenize(song, msgToToken))
        title = filename[18:-4] + "TEMPOFIX"
        make_midi(tokenizedSongs[0], tokenToMsg, title)
    # I return len(uniques) instead of tokenizedUniques because it's just a list [0 .. len(uniques) - 1]
    return 0


folder = "Generated"
title = "Practice1Fixed_"+folder

tkMsgsList, numUniques, tokenToMsg, msgToToken = fixSongs(folder)
