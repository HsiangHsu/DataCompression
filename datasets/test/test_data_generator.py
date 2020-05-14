import numpy as np

element_0 = [
    [0, 5, 1, 1],
    [0, 0, 1, 1],
    [3, 4, 3, 3],
    [3, 3, 3, 3]
]

element_1 = [
    [0, 0, 2, 2],
    [0, 0, 2, 6],
    [2, 2, 3, 3],
    [2, 2, 3, 3]
]

element_2 = [
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [3, 3, 9, 9],
    [3, 3, 9, 9]
]

element_3 = [
    [2, 2, 3, 3],
    [2, 2, 3, 3],
    [2, 2, 9, 9],
    [2, 2, 9, 9]
]

data = np.array([element_0, element_1, element_2, element_3], dtype=np.uint8)
print(data)

with open('test_data.np', 'wb') as f:
    np.save(f, data)
