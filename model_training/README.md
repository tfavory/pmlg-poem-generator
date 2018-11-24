# Train your own poem generator model

There are notebooks of interest here:
 * A poem generator called *poem_model*. It trains a model that generates text backwards (from the last letter to the first). The idea is to use it with a rhyme generator.
 * A rhyme generator called *rhymes_model*. It trains a rhyme generator supposed to return lists of words that rhyme
 
The first notebook relies on a corpus of poems named 'corpus.txt'

The second notebook will take 'rhymes.txt' as a database of rhymes.

Note that 'rhymes.txt' is nothing but the rhymes extracted from 'corpus.txt', with the help of corpus2rhymes.py. 

Note also that corpus2rhymes.py performs some preliminary corpus cleaning as well. Note that it is not perfect.
