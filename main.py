import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# Imagine line = winning ticket = ticket
tickets_grouped = {}

with open("./lotto.csv") as file:
    for line in file:
        data = line.strip().split("\n")
        ticket = "".join(data).split(",")

        ticket_date = ticket[1:2]
        ticket_numbers = ticket[2:]

        tickets_grouped["".join(ticket_date)] = ticket_numbers


def prepare_data(data, look_back):
    X, y = [], []
    for x in range(len(data) - look_back):
        X.append(data[x : x + look_back])
        y.append(data[x + look_back])
    return np.array(X), np.array(y)


previous_winners = []

for date, numbers in tickets_grouped.items():
    previous_winners.append(numbers)

data = np.array(previous_winners, dtype=np.float32)
"""
look_back = 5 ???

X, y = prepare_data(data, look_back)

model = Sequential([LSTM(50), Input(shape=(look_back, 6)), Dense(6)])
model.compile(optimizer="adam", loss="mse")

model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

last_sequence = previous_winners[:-1]
prediction = model.predict(last_sequence)

print(prediction[0])
"""
