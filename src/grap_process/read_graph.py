import numpy as np

data = np.load("philly_zip_graph_2022.npz")

adj = data["adjacency"]
zipcodes = data["zipcodes"]
homeless_count = data["homeless_count"]