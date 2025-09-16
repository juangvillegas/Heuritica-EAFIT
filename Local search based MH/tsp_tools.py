import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import pandas as pd
import plotly.express as px


def read_data(dataFile):
    f_entrada=open(dataFile,"r")
    #We read the first three lines without extranting any data
    a=f_entrada.readline()
    a=f_entrada.readline()
    a=f_entrada.readline()
    #We read the fourth line and extract the size of the instance
    a=f_entrada.readline()
    b=a.split()
    n=int(b[1])
    #We read two other comment lines without extranting any data
    a=f_entrada.readline()
    a=f_entrada.readline()
    #Then we read the coordinates of all the customers
    coords=[]
    for i in range(n):
      a=f_entrada.readline()
      b=a.split()
      row=[]
      row.append(float(b[1]))
      row.append(float(b[2]))
      coords.append(row)
    return(coords)


# calculates a arc distance
def calculate_distances(coords):  
  dist=distance.cdist(coords,coords, metric='euclidean')
  return dist

# Calculates total distance
def tsp_distance(d, # Distance matrix
tour): # List of cities to order
  n = len(tour)
  length = 0 # Tour length
  for i in range(1, n): # Cities from 0 to n -1
    dist=d[tour[i-1]][tour[i]]
    #print(str(i) + " "+ str(dist))
    length += dist # distance from a city and its predecessor

  dist=d[tour[n-1]][tour[0]]
  length += dist

  return length

# Scatter plot of the points of a solution
def print_TSPdata(c):
  
  #Extracting the data from the coordinates
  data = np.array(c)
  x, y = data.T
  # plot our list in X,Y coordinates
  plt.scatter(x, y)
  plt.show()

# Plot the route of a given solution
def plot_tsp_route(cities, solution):
    # Extract the coordinates of cities in the order specified by the solution
    ordered_cities = [cities[i] for i in solution]
    ordered_cities.append(ordered_cities[0])  # Complete the loop

    x, y = zip(*ordered_cities)  # Split coordinates into x and y

    # Plot the TSP route
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', linestyle='-', markersize=8, markerfacecolor='r')

    # Annotate the cities with their indices
    # for i, (xi, yi) in enumerate(zip(x, y)):
    #     plt.annotate(str(solution[i]), (xi, yi), fontsize=12, ha='center', va='bottom')

    plt.title("TSP Route Visualization")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.grid(True)  
    plt.show()


def plot_trace(trace):
  df = pd.DataFrame(trace, columns=['iteration', 'current', 'best'])
  fig = px.line(df, x='iteration', y=['current', 'best'], title='Search trajectory')
  
  return fig
