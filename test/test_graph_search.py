from dataloaders.complete_dataloader import graph_search
import numpy as np

mat = np.array([[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 0]])

result = graph_search(1, mat, 6)

print(result)
