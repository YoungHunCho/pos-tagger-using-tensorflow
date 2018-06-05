from konlp.kma.klt import klt
k = klt.KltKma()

file = open("corpus.some", "r")

tagged = open("tagged.some", "w")
while(True):
    line = file.readline()
    if not line: break
    
    for words in k.analyze(line):
        word = ""
        tag = ""
        word = words[0]

        for tags in words[1]:
            tag += tags[1] + "_"
        tag = tag[:-1]

        tagged.write(word + "/" + tag + " ")
    tagged.write("\n")

tagged.close()