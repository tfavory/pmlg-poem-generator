# pmlg-poem-generator

This is a poem generator trained with Gated Recurrent Units (GRU) on a corpus of poems including those in:
* [Poems Every Child Should Know](http://www.gutenberg.org/ebooks/16436), by Mary E. Burt
* [Poems](http://www.gutenberg.org/ebooks/52546), by Thomas Hall Shastid
* [Poems of Progress and New Thought Pastels](http://www.gutenberg.org/ebooks/3228), by Ella Wheeler Wilcox
* [Poems Teachers Ask For](http://www.gutenberg.org/ebooks/18909)
* [Poems Teachers Ask For, Book Two](http://www.gutenberg.org/ebooks/19469)
* [The Pied Piper of Hamelin, and Other Poems](http://www.gutenberg.org/ebooks/42850), by Robert Browning

Most of the code comes from [Tensorflow's text generation tutorial](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/eager/python/examples/generative_examples/text_generation.ipynb).

Please refer to the notebook for an overview of the results.

## Improvement ideas

### Short term:
* Find more data
* Train a model based on sequences of words instead of characters (memory issues)

### Middle term goals:
* Try Bi-directional models (failed to train a model conform to [Tensorflow's text generation tutorial](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/eager/python/examples/generative_examples/text_generation.ipynb).)

The idea is to generate the rhymes first, and then write the lines from the last word to the first. This process makes sure that the generated poems rhyme.

### Long term goals
* Style transfert
* Generate other kind of text (Haiku, magazines, articles, books...)
