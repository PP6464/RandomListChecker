from random import randint, choices
import numpy as np

# # Generate the computer random data
# with open("computer_random.txt", "a") as computer:
#     for _ in range(10000):
#         length = randint(1, 100)
#         rand_list = []
#         for i in range(length):
#             rand_list.append(str(randint(1, 100)))
#         computer.write(f"[{','.join(rand_list)}]\n")


# ALTERNATE WAY TO GENERATE HUMAN RANDOM DATA
# When asked to choose a random number between 1 and 100, most humans choose 37 or 73
# Thus can model P(X) = N(37, 100) + N(73, 100) (i.e. sum P(X) from the dists., where X is a random number 1<=X<=100)
# The length of the list follows L~N(15, 100)
def L(x: int) -> float:  # The length normal distribution
    return np.exp(-((x - 15) ** 2) / 100)


def P(x: int) -> float:  # The random int choosing normal distribution
    return np.exp(-((x-37)**2)/100) + np.exp(-((x-73)**2)/100)


def rand_length() -> int:
    options = [k + 1 for k in range(100)]
    weights = [L(k + 1) for k in range(100)]
    return choices(options, weights=weights)[0]


def rand_int() -> int:
    options = [k + 1 for k in range(100)]
    weights = [P(k + 1) for k in range(100)]
    return choices(options, weights=weights)[0]


with open("human_random.txt", "a") as human:
    for _ in range(10000):
        length = rand_length()
        rand_list = []
        for i in range(length):
            rand_list.append(str(rand_int()))
        human.write(f"[{','.join(rand_list)}]\n")
