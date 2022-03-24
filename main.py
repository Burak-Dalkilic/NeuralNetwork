import numpy as np
import pickle, gzip
import solution as sol
import sys

# Read in training data
with gzip.open(sys.argv[1]) as f:
    data, labels = pickle.load(f, encoding='latin1')
    data = sol.magic_function(data)

# Starting values for w and b
w0 = np.zeros(data.shape[1])
b0 = 0

# Optimization
w,b = sol.minimize_loss(data, labels, w0,b0, sys.argv[3:])

# Test on test data
with gzip.open(sys.argv[2]) as f:
    test_data, test_labels = pickle.load(f, encoding='latin1')
    test_data = sol.magic_function(test_data)

yhat = sol.f(test_data, w, b)[0]>=.5
print(np.mean(yhat==test_labels)*100, "% of test examples classified correctly.")
