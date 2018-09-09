# pmlg-poem-generator

This is a poem generator trained with Gated Recurrent Units (GRU) on a corpus of poems including those in:
* Poems every child should know
* Some other poems
* And some more poems

Most of the code comes from [Tensorflow's text generation tutorial](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/eager/python/examples/generative_examples/text_generation.ipynb).

Please refer to the notebook for an overview of the results.

## To do next

### Short term:
* Clean the corpus
* Find more data
* Try LSTM
* Train a model based on sequences of words instead of characters

### Middle term goals:
* Create a rhyme generator
* Train a model backward: from the last word to the first

The idea is to generate the rhymes first, and then write the lines from the last word to the first. This process makes sure that the generated poems rhyme.

* Learn to count syllables?

### Long term goals
* Generate other kind of text (Haiku, magazines, articles, books...)
