import numpy as np
from random import randint
from math import ceil, floor
from random import shuffle



given = get_k_fold_inds(100, 7)
print(list(given).count(1))
print(list(given).count(2))
print(list(given).count(3))
print(list(given).count(4))
print(list(given).count(5))
print(list(given).count(6))
print(list(given).count(7))
print(len(given))
print(given)