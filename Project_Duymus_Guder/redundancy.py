import graph

"""
    Calculates the redundancy of a layer using quotient space size and l-covering number estimation.
    l-covering number is calculated as an average of 1-cover and 2-cover as in the paper.
    w1 is a probability weight that balances the importance of qss and l-covering number.
"""
def redundancy(graph, w1):
  estCover = (graph.lcover(graph, 1) + graph.lcover(graph, 2)) / 2
  k = graph.quotientSpaceSize(graph)
  return graph.shape[0] / (w1 * k + (1 - w1) * estCover) # N / (w_1 * k + w_2 * N_1^c)