import string


corpus = open('poem_corpus.txt')
new_corpus = open('corpus.txt', 'w')
rhymes = open('rhymes.txt', 'w')
c = 1
corpus.readline()

blank_count = 3
for line in corpus.readlines():
    line = line.lstrip()
    line = line.rstrip(string.punctuation + string.whitespace)
    
    if line == '':
        blank_count += 1
        if blank_count < 2:
            print(line.lower())
            new_corpus.write(line.lower() + '\n')
    else:
        if blank_count <= 2:
            istitle = line[-1] in string.ascii_uppercase
            isauthor = line[0] in string.punctuation 
            if len(line) >= 5 and not istitle and not isauthor:
                last_word = line.split()[-1]
                rhymes.write(last_word + '\n')
                print(line.lower())
                new_corpus.write(line.lower() + '\n')
        blank_count = 0
    c += 1

corpus.close()
new_corpus.close()
rhymes.close()
