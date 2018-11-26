# pmlg-poem-generator

This is a poem generator trained with Gated Recurrent Units (GRU) on a corpus of poems including:
* [Poems Every Child Should Know](http://www.gutenberg.org/ebooks/16436), by Mary E. Burt
* [Poems](http://www.gutenberg.org/ebooks/52546), by Thomas Hall Shastid
* [Poems of Progress and New Thought Pastels](http://www.gutenberg.org/ebooks/3228), by Ella Wheeler Wilcox
* [Poems Teachers Ask For](http://www.gutenberg.org/ebooks/18909)
* [Poems Teachers Ask For, Book Two](http://www.gutenberg.org/ebooks/19469)
* [The Pied Piper of Hamelin, and Other Poems](http://www.gutenberg.org/ebooks/42850), by Robert Browning

Most of the code comes from [Tensorflow's text generation tutorial](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/eager/python/examples/generative_examples/text_generation.ipynb).

Please refer to the notebook for an overview of the results.

## How to generate poetry?

Use the pretrained model called poem_generator.

 - In the same folder, save the notebook called poem_generator.ipynb and the 2 npy files: model_poems.npy, and model_rhymes.npy 
 - Run poem_generator.ipynb
 
Disclaimer: Saving the model with npy files is probably less efficient than loading a Checkpoint, but I did not manage to do it


## How to train your own model

(optional) if you want to train a model on your own text files:

 - Find some text, called the file 'poem_corpus.txt'
 - Run corpus_cleaner.py. It should create two text files: 'corpus.txt', and 'rhymes.txt'.
 
To train a model:

 - Run 'rhyme_model.ipynb'. It creates 6 npy files that store the weigths of your rhyme model.
 - Run 'poem_model.ipynb'
 - Run 'poem_generator.ipynb'
 
 
## Improvement ideas

### Short term:
* Update weights with another corpus
* Try data from [GutenTag](http://www.cs.toronto.edu/~jbrooke/gutentag/)
* Train a model based on sequences of words instead of characters (memory issues)

### Middle term goals:
* Try Bi-directional models (failed to train a model conform to [Tensorflow's text generation tutorial](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/eager/python/examples/generative_examples/text_generation.ipynb).)

### Long term goals
* Style transfert
* Generate other kind of text (Haiku, magazines, articles, books...)
