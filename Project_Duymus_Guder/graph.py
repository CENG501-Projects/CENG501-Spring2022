import numpy as np

"""
    Creation of the graph of a network layer. Gamma decides the sparsity. 
    As gamma increases, the similarity threshold for an edge decreases, allowing a denser graph.
"""
def constructGraph(layer, gamma):
  noFilters = layer.out_channels
  graph = np.zeros(shape=(noFilters, noFilters))

  for i in range(noFilters):
    filter1 = layer.weight[i]
    for j in range(noFilters):
      if i == j: continue
      filter2 = layer.weight[j]

      if graph[j][i]:
        graph[i][j] = 1
        continue

      # flatten
      filter1, filter2 = filter1.reshape(-1, 1), filter2.reshape(-1, 1)
      n = filter1.shape[0]  # n = N_i * h_i * w_i

      # normalize
      filter1 = filter1 / np.linalg.norm(filter1, ord=2)
      filter2 = filter2 / np.linalg.norm(filter2, ord=2)

      # distance
      dist = np.linalg.norm(filter1-filter2, ord=2)
      if (dist / np.sqrt(n)) <= gamma:
        graph[i][j] = 1

  return graph

"""
    Helper function for Quotient space size analysis.
"""
def QSSUtility(graph, visited, idx):
  n = graph.shape[0]
  visited[idx] = 1

  for i in range(n):
    if(graph[idx][i] == 1 and visited[i] == 0):
      QSSUtility(graph, visited, i)

"""
    Quotient space size analysis.
    Uses a logic similar to DFS to return the number of connected components. 
    Count is increased by one when a connected component is fully traversed.
"""
def quotientSpaceSize(graph):
  n = graph.shape[0]
  visited = np.zeros(n)
  count = 0

  for i in range(n):
    if(visited[i] == 0):
      QSSUtility(graph, visited, i)
      count = count + 1
  return count

"""
    Returns the index of the node that has the highest degree and not yet covered.
    Used in lcover function.
"""
def maxNode(graph, covered):
  n = graph.shape[0]
  node = 0
  degree = 0

  for i in range(n):
    if(covered[i] == 0):
      currentDegree = np.sum(graph[i, :])
      if(currentDegree >= degree):
        degree = currentDegree
        node = i
  return node

"""
    Calculates and returns an estimation to l-covering number of the graph.
"""
def lcover(graph, l):
  n = graph.shape[0]
  covered = np.zeros(n)
  count = 0

  if(l == 1):
    dist = graph
  elif(l == 2):
    dist = np.dot(graph, graph.T) # 2-distance matrix

  while(np.sum(covered) < n):
    count = count + 1
    max = maxNode(graph, covered)
    covered[max] = 1
    for i in range(n):
      if(dist[max][i] > 0):
        covered[i] = 1
  return count