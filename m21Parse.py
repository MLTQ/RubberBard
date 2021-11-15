import music21 as m21

#m21.converter.parse("tinynotation: 3/4 c4 d8 f g16 a g f#").show()

n = m21.note.Note("D#3")
n.duration.type = 'half'
n.show()