# implement tutorial from
# http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

"""
Sections
1. Get data
2. Clean data
3. Transform to one hot

"""
import numpy as np
import keras
import re
from sklearn.preprocessing import LabelEncoder


limit_data = 10**4


class WordToVec:
    def __init__(self, num_dim: int):
        self.num_dim = num_dim

    def fit(self, X: str, batch_size: int, epochs: int,
            window_size: int) -> None:
        """X is the string to be trained on"""
        self.X = X
        self.window_size = window_size
        self._clean_data()
        self._one_hot_encoder()
        self._create_model()
        self._make_samples()
        self.model.fit(self.X, self.y, batch_size, epochs)

    def _clean_data(self) -> None:
        self.words = self.X.split()[:limit_data]
        self.words = [x.lower() for x in self.words]  # all to lowercase

        regex = re.compile('[^a-zA-Z]')
        # strip all non-letter
        self.words = [regex.sub('', x) for x in self.words]

    def _one_hot_encoder(self) -> None:
        self.words = np.array(self.words)
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(self.words)
        self.one_hot_encoded = keras.utils.to_categorical(integer_encoded)

    def _create_model(self) -> None:
        input_dim = self.one_hot_encoded.shape[1]
        print(self.num_dim, input_dim)
        self.model = keras.Sequential([
            keras.layers.Dense(self.num_dim, input_shape=(input_dim,)),
            keras.layers.Activation('linear'),
            keras.layers.Dense(input_dim),
            keras.layers.Activation("softmax")
        ])
        self.model.compile(optimizer="rmsprop", loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def _make_samples(self) -> None:
        size = self._calculate_num_samples()
        dim = self.one_hot_encoded.shape[1]
        self.X = np.empty((size, dim))
        self.y = np.empty((size, dim))

        displacement = 0
        for i in range(1, self.window_size+1):
            current_y = np.roll(self.one_hot_encoded, i, axis=0)[i:]
            current_x = self.one_hot_encoded[:-i]

            if current_x.shape != current_y.shape:
                raise ValueError("Something Wrong")

            self.X[displacement:current_x.shape[0] + displacement] = current_x
            self.y[displacement:current_x.shape[0] + displacement] = current_x

            displacement += current_x.shape[0]

            current_y = np.roll(self.one_hot_encoded, -1, axis=0)[:-i]
            current_x = self.one_hot_encoded[i:]

            if current_x.shape != current_y.shape:
                raise ValueError("Something Wrong")

            self.X[displacement:current_x.shape[0] + displacement] = current_x
            self.y[displacement:current_x.shape[0] + displacement] = current_x

            displacement += current_x.shape[0]

    def _calculate_num_samples(self) -> int:
        """Calculates the number of training samples based on the size of
        window and number of words"""
        num_samples = self.one_hot_encoded.shape[0]
        window_size = self.window_size
        return window_size*(2*num_samples-window_size-1)


# Get data, Sherlock Holmes Book
with open("big.txt", "r") as file:
    data = file.read()

word2vec = WordToVec(300)
word2vec.fit(data, 252, 1000, 1)


