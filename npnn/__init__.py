"""Numpy Neural Network implementation for 18-661

     __      ___       __       __
  //   ) ) //   ) ) //   ) ) //   ) )
 //   / / //___/ / //   / / //   / /
//   / / //       //   / / //   / /

Performance
-----------
When using the settings described in problem 3.4 (256-64-10 units, SGD),
our reference implementation takes approximately 2.5 seconds to train one epoch
on an AMD Ryzen 7 5800X, and 6.3 seconds on an Intel i7-1167G7.

You can get a rough estimate of how long it should take on your machine by
looking up the PassMark (https://www.passmark.com/) score of your CPU, and
comparing it to our reference implementation. Assuming linear scaling, your
runtime should be roughly `70000 / passmark`.

If your implementation is significantly (>2x) slower than the reference
implementation, check to see if all forward and backward passes are fully
vectorized.
"""

from .modules import Variable, ELU, Dense, SoftmaxCrossEntropy, Flatten
from .model import Sequential
from .optimizer import SGD, Adam
from .dataset import Dataset, load_mnist

__all__ = [
    "Variable", "ELU", "Dense", "SoftmaxCrossEntropy", "Flatten",
    "Sequential",
    "SGD", "Adam",
    "Dataset", "load_mnist"
]
