from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from layers.Attention import Attention
from tensorflow.keras.models import load_model
from random import randint

saved_model = load_model("model.keras", custom_objects={"Attention": Attention})

rand_list = []

for i in range(randint(1, 100)):
    rand_list.append(randint(1, 100)/100)

print(saved_model(np.array(pad_sequences([rand_list], maxlen=100))))
