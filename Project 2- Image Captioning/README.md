## Project Overview

In this project, you will create a neural network architecture to automatically generate captions from images.

After using the Microsoft Common Objects in COntext [(MS COCO) dataset](http://cocodataset.org/#home) to train your network, you will test your network on novel images!




<p align="center">
<img width="1354" alt="Image Captioning Model" src="https://user-images.githubusercontent.com/33187812/68330571-27dfb980-00dc-11ea-96c7-6389d020145a.png"/>
<b>Image Captioning Model</b>
</p>


### LSTM Inputs/Outputs

#### LSTM Decoder
In the project, we pass all our inputs as a sequence to an LSTM. A sequence looks like this: first a feature vector that is extracted from an input image, then a start word, then the next word, the next word, and so on!

#### Embedding Dimension
The LSTM is defined such that, as it sequentially looks at inputs, it expects that each individual input in a sequence is of a consistent size and so we embed the feature vector and each word so that they are embed_size.

#### Sequential Inputs
So, an LSTM looks at inputs sequentially. In PyTorch, there are two ways to do this.

The first is pretty intuitive: for all the inputs in a sequence, which in this case would be a feature from an image, a start word, the next word, the next word, and so on (until the end of a sequence/batch), you loop through each input like so:

```
for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)
```

The second approach, which this project uses, is to give the LSTM our entire sequence and have it produce a set of outputs and the last hidden state:

```
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state

# Add the extra 2nd dimension
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
```
