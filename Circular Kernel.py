import numpy as np

r = 12
kernel = np.fromfunction(lambda x, y: ((x-r/2)**2 + (y-r/2)**2 <= (r/2)**2)*1, (r+1, r+1), dtype=int).astype(np.uint8)
print(kernel)
