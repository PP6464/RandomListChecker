import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import GRU, LSTM, Dense, Input, Masking, Bidirectional
from sklearn.model_selection import train_test_split
from model.layers.Attention import Attention


# Parses "[1,2,3]" and returns a list [1,2,3]
def parse_list_ints(text: str) -> list[int]:
    text_chars = list(text.strip())
    # Remove square brackets
    text_chars.pop()
    text_chars.pop(0)
    # Return text as a list (remove commas)
    return list(map(lambda x: int(x), "".join(text_chars).split(",")))


with open("../data/computer_random.txt", "r") as computer_file:
    computer_data = list(map(parse_list_ints, computer_file.readlines()))

with open("../data/human_random.txt", "r") as human_file:
    human_data = list(map(parse_list_ints, human_file.readlines()))

human_labels = [0] * 10000
computer_labels = [1] * 10000

data = human_data + computer_data
labels = human_labels + computer_labels

# Split data into training and test data sets
data = pad_sequences(data, maxlen=100, padding="post")  # pad lists to uniform length for RNN
data = list(map(lambda x: x / 100, data))  # Normalise data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build the GRU (Gated Recurrent Units) and LSTM (Long Short-term Memory) RNN
seq_input = Input(shape=(100, 1))  # Input layer
masked_input = Masking()(seq_input)  # Mask padded values
gru_output = Bidirectional(GRU(128, return_sequences=True))(masked_input)
lstm_output = Bidirectional(LSTM(128, return_sequences=True))(gru_output)

attention_layer = Attention(64)

context_vector, attention_weights = attention_layer(lstm_output, lstm_output[:, -1, :])

output = Dense(1, activation="sigmoid")(context_vector)  # Binary classification

model = Model(inputs=seq_input, outputs=output)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

model.fit(x_train, np.array(y_train), epochs=10, batch_size=25, validation_split=0.2)

loss, accuracy = model.evaluate(x_test, np.array(y_test))

saved_model = load_model("model.keras")

saved_model_loss, saved_model_accuracy = saved_model.evaluate()

if saved_model_accuracy < accuracy:
    model.save("model.keras")

print(f"Accuracy: {accuracy:.4f}")
