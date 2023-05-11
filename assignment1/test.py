import numpy as np

from cs231n.classifiers import KNearestNeighbor

# Create a kNN classifier instance. 
# Remember that training a kNN classifier is a noop: 
# the Classifier simply remembers the data and does no further processing 
classifier = KNearestNeighbor()
classifier.train(np.random.random((5000, 3072)), np.random.randint(10, size=5000))

dists = classifier.compute_distances_two_loops(np.random.random((500, 3072)))
print(dists.shape)
dists