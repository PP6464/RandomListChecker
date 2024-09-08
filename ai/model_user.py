from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
from random import randint

with open("model.pickle", "rb") as model:
    saved_model = pickle.load(model)

    rand_list = []

    for i in range(randint(1, 100)):
        rand_list.append(randint(1, 100)/100)

    print(saved_model(pad_sequences([np.array(rand_list)], maxlen=100, padding="post")))
