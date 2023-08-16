import datetime
import matplotlib.pyplot as plt
import numpy as np

date = np.array('2020-10-09', dtype=np.datetime64)
date = date + np.arange(30)
date = date.astype("datetime64[ns]")
print(date)

numbers = np.array(1, dtype=np.int)
numbers = numbers + np.arange(30)
numbers = numbers.astype("float64")
print(numbers)


print(date.dtype)
print(numbers.dtype)
print(type(date))
print(type(numbers))

plt.plot(date, np.array(numbers, dtype=np.float64))
plt.plot(date, np.array(numbers, dtype=np.float64))
plt.show()
















