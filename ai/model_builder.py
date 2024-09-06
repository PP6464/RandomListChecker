import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, InputLayer, Masking, Bidirectional
from sklearn.model_selection import train_test_split


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

human_labels = [0] * 1000
computer_labels = [1] * 1000

data = human_data + computer_data
labels = human_labels + computer_labels

# Split data into training and test data sets
data = pad_sequences(data, maxlen=100, padding="post")  # pad lists to uniform length for RNN
data = list(map(lambda x: x/100, data))  # Normalise data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build the GRU (Gated Recurrent Units) RNN
model = Sequential()

model.add(InputLayer(shape=(100, 1)))
model.add(Masking())  # Masking layer to remove padded values
model.add(Bidirectional(GRU(64, return_sequences=True)))  # Bidirectional lets the RNN learn both about things previous and ahead in the list
model.add(Bidirectional(GRU(64)))  # Bidirectional lets the RNN learn both about things previous and ahead in the list
model.add(Dense(1, activation="sigmoid"))  # Output layer

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

model.fit(x_train, np.array(y_train), epochs=25, batch_size=10, validation_split=0.2)

loss, accuracy = model.evaluate(x_test, np.array(y_test))

saved_model = load_model("model.keras")  # Load saved model for comparison

saved_model_loss, saved_model_accuracy = saved_model.evaluate(x_test, np.array(y_test))

if saved_model_accuracy < accuracy:
    model.save("model.keras")

print(f"Accuracy: {accuracy:.4f}")
