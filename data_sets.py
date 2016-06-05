import numpy


class Data_set:
    def __init__(self, X, Y):
        self._images = X.values
        self._labels = Y.values
        self._num_examples = len(X)
        self._index_in_epoch = 0
        self._epochs_completed = 0

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


class Data_sets:
    def __init__(self, train, validation, test):
        self.train = train
        self.validation = validation
        self.test = test
