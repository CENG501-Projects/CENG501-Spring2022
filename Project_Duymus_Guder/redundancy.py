from graph import *

"""
    Calculates the redundancy of a layer using quotient space size and l-covering number estimation.
    l-covering number is calculated as an average of 1-cover and 2-cover as in the paper.
    w1 is a probability weight that balances the importance of qss and l-covering number est.
"""
def redundancy(graph, w1):
  estCover = (lcover(graph, 1) + lcover(graph, 2)) / 2
  k = quotientSpaceSize(graph)
  return graph.shape[0] / (w1 * k + (1 - w1) * estCover) # N / (w_1 * k + w_2 * N_1^c)