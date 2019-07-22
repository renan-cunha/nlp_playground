import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity


class WordToVec:
    def __init__(self, matrix: np.array, words: List[str]):
        """The first dimension of the array is the words, the second is the
        vector dimensions"""
        self.__matrix = matrix
        self.__words = words

    def __get_vector(self, word: str) -> np.ndarray:
        index = self.__words.index(word)
        return self.__matrix[index]

    def __get_word(self, vector: np.ndarray) -> str:
        index = np.argwhere(self.__matrix == vector)[0]
        return self.__words[index]

    def find_similarity(self, word: str, size: int = 1) -> List[str]:
        """Returns the most similar words with input word"""
        word_vec = np.empty_like(self.__matrix)
        word_vec[:] = self.__get_vector(word)
        cosine_results = cosine_similarity(self.__matrix, word_vec)
        cosine_results = np.argsort(cosine_results)[1:size+1]




