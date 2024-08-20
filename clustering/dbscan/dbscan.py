import numpy as np
import random
import os
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator

# Configs
epsilon = 10
min_points = 4
data_generator_set_count = 4
data_generator_data_for_each_count_count = 20
data_generator_bounds = [(10, 30), (40, 60), (70, 90), (0, 100)]
# data_generator_bounds = [(0, 100), (0, 100), (0, 100)]
storage_path = './out'

# Create a path to storage the result
if not os.path.exists(storage_path):
    os.mkdir(storage_path)

# Create random points in the flat, each cluster contains 40 points
# index 0 is for x-axis
# index 1 is for y-axis
# index 2 is for centroid and note that it is just used in clustering, initial is meaningless
data_size = (data_generator_data_for_each_count_count, 3)
data_sets = []

# Create three crossed cluster
for i in range(data_generator_set_count):
    data_sets.append(np.random.randint(data_generator_bounds[i][0], data_generator_bounds[i][1], size=data_size))
data_set = data_sets[0]
for i in range(data_generator_set_count - 1):
    data_set = np.concatenate((data_set, data_sets[i + 1]), axis=0)
for data in data_set:
    data[2] = -1

# Create origin plt
plt.scatter(data_set[:, 0], data_set[:, 1], s=20)
locator = MultipleLocator(5)
gca = plt.gca()
gca.xaxis.set_major_locator(locator)
gca.yaxis.set_major_locator(locator)
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid()
plt.title("Random points without clustering")
plt.savefig("./out/dbscan_origin.png", dpi=300)
# plt.show()
plt.close()


# Calculate the Euclidean distance of two point
def calculate_distance(point1, point2):
    return float(np.sqrt(np.square(np.abs(point1[0] - point2[0])) + np.square(np.abs(point1[1] - point2[1]))))


label = 1
for index in range(len(data_set)):
    if data_set[index][2] != -1:
        continue
    data_set[index][2] = label
    cluster_indexes = [index]
    cluster_count = 0
    while len(cluster_indexes) != cluster_count:
        cluster_count = len(cluster_indexes)
        for cluster_index in cluster_indexes:
            for data_set_index in range(len(data_set)):
                if cluster_index == data_set_index:
                    continue
                if data_set[data_set_index][2] != -1:
                    continue
                distance = calculate_distance(data_set[cluster_index], data_set[data_set_index])
                if distance <= epsilon:
                    cluster_indexes.append(data_set_index)
                    data_set[data_set_index][2] = label
    if len(cluster_indexes) < min_points:
        for index_reset in cluster_indexes:
            data_set[index_reset][2] = 0
        continue
    label += 1

#
data_set = data_set[np.lexsort(data_set.T)]
print(data_set)
cluster_labels = list(set(data_set[:, 2]))
print(cluster_labels)
legends = []
for cluster_label in cluster_labels:
    x = []
    y = []
    for data in data_set:
        if data[2] == cluster_label:
            x.append(data[0])
            y.append(data[1])
    if cluster_label == 0:
        plt.scatter(x, y, s=20, marker='*')
        legends.append("Noise")
    else:
        plt.scatter(x, y, s=20)
        legends.append("Cluster " + str(cluster_label))

locator = MultipleLocator(5)
gca = plt.gca()
gca.xaxis.set_major_locator(locator)
gca.yaxis.set_major_locator(locator)
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.title("Clustered by DBSCAN")
plt.legend(legends)
plt.grid()
plt.savefig("./out/dbscan_result.png", dpi=300)
# plt.show()
plt.close()
