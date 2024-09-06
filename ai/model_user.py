from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
from random import randint

saved_model = load_model("model.keras")

rand_list = []

for i in range(randint(1, 100)):
    rand_list.append(randint(1, 100)/100)

print(saved_model(pad_sequences([np.array(rand_list)], maxlen=100, padding="post")))
