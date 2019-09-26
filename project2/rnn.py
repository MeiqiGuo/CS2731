"""
An RNN model for character-level language model for predicting next character in each sentence.
Reference to https://www.tensorflow.org/tutorials/sequences/text_generation.
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import os
import time

import utils
import argparse


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
                                 rnn(rnn_units, return_sequences=True, recurrent_initializer='glorot_uniform', stateful=True),
                                 tf.keras.layers.Dropout(0.2),
                                 rnn(rnn_units, return_sequences=True, recurrent_initializer='glorot_uniform', stateful=True),
                                 tf.keras.layers.Dropout(0.2),
                                 tf.keras.layers.Dense(vocab_size)])
    return model


def loss(labels, logits):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)


def predict(model, test_data):
    y_pred = []
    y_true = []
    for inputs, targets in test_data:
        # each sentence(line) in test dataset
        model.reset_states()
        for input, target in zip(inputs, targets):
            input_eval = tf.expand_dims([input], 0)
            prediction = model(input_eval)
            prediction = tf.squeeze(prediction, 0).numpy()
            predicted_id = np.argmax(prediction)
            y_pred.append(predicted_id)
            y_true.append(target)
    acc = sum(np.array(y_pred) == np.array(y_true)) * 1.0 / len(y_pred)
    print("The accuracy of predicting the next character is {}.".format(acc))
    return acc, y_pred, y_true


parser = argparse.ArgumentParser()
parser.add_argument("trainFileName", help="file name of the training dataset")
parser.add_argument("testFileName", help="file name of the test dataset")
parser.add_argument("vocabFileName", help="file name of the vocabulary dictionary")
parser.add_argument("checkpoint_dir", help="file name of the checkpoint directory")
parser.add_argument("embedding_dim", help="the embedding dimension", type=int)
parser.add_argument("rnn_units", help="the number of RNN units", type=int)
parser.add_argument("epochs", help="the number of epochs", type=int)
parser.add_argument("-load_previous", help="if true, load the most recently saved model", action='store_true')
args = parser.parse_args()

# Load vocab
vocab = utils.load_vocab(args.vocabFileName)

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = args.embedding_dim

# Number of RNN units
rnn_units = args.rnn_units

# Use CuDNNGRU if running on GPU.
if tf.test.is_gpu_available():
    rnn = tf.keras.layers.CuDNNLSTM
else:
    import functools
    rnn = functools.partial(tf.keras.layers.LSTM, recurrent_activation='sigmoid')


if args.epochs:  # if epochs is not 0, then train the model; otherwise, directly predict on test data
    # Load train data
    text = utils.load_train_data(args.trainFileName)
    idx2char = {i: w for w, i in vocab.items()}
    text_as_int = utils.text2id(text, vocab)

    # The maximum length sentence we want for a single input in characters
    seq_length = 100
    examples_per_epoch = len(text)//seq_length

    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
    dataset = sequences.map(utils.split_input_target)
    # Print the first examples input and target values
    for input_example, target_example in dataset.take(1):
        print ('Input data: ', ''.join([idx2char[x] for x in input_example.numpy()]))
        print ('Target data:', ''.join([idx2char[x] for x in target_example.numpy()]))

    # Batch size
    BATCH_SIZE = 64
    steps_per_epoch = examples_per_epoch//BATCH_SIZE

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 10000
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


    # Build model
    model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)
    if args.load_previous:
        model.load_weights(tf.train.latest_checkpoint(args.checkpoint_dir))
    print(model.summary())

    # Check the shape of the output
    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    model.compile(optimizer=tf.train.AdamOptimizer(), loss=loss)

    # Config check points
    # Name of the checkpoint files
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    checkpoint_prefix = os.path.join(args.checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)


    # Train the model
    history = model.fit(dataset.repeat(), epochs=args.epochs, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])

# Prediction on test

# Change model's batch size to 1
#old_model = tf.keras.models.load_model(os.path.join(args.checkpoint_dir, "ckpt_10"))
#weights_file = os.path.join(args.checkpoint_dir, "weights_ckpt_10")
#old_model.save_weights(weights_file)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(args.checkpoint_dir))
#model.load_weights(weights_file)
model.build(tf.TensorShape([1, None]))
print(model.summary())

# Process text data
test_text = utils.load_test_data(args.testFileName)
test_as_int = utils.test2id(test_text, vocab)
test_data = list(map(utils.split_input_target, test_as_int))
print("Example of test data {}".format(test_data[0]))
acc, y_pred, y_true = predict(model, test_data)



