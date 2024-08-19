import numpy as np
import random
import os
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator

# Configs
loop_limit = 1000
centroids_count = 3
data_generator_count = 3
data_generator_bounds = [(10, 60), (25, 75), (40, 90)]
# data_generator_bounds = [(0, 100), (0, 100), (0, 100)]
storage_path = './out'

# Create a path to storage the result
if not os.path.exists(storage_path):
    os.mkdir(storage_path)

# Create random points in the flat, each cluster contains 40 points
# index 0 is for x-axis
# index 1 is for y-axis
# index 2 is for centroid and note that it is just used in clustering, initial is meaningless
data_size = (40, 3)
data_sets = []
# Create three crossed cluster
for i in range(data_generator_count):
    data_sets.append(np.random.randint(data_generator_bounds[i][0], data_generator_bounds[i][1], size=data_size))
data_set = data_sets[0]
for i in range(data_generator_count - 1):
    data_set = np.concatenate((data_set, data_sets[i + 1]), axis=0)
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
plt.savefig("./out/kmeans_origin.png")
# plt.show()
plt.close()


# Calculate the Euclidean distance of two point
def calculate_distance(point1, point2):
    return float(np.sqrt(np.square(np.abs(point1[0] - point2[0])) + np.square(np.abs(point1[1] - point2[1]))))


# Random find three points as centroids
centroids = random.sample(data_set.tolist(), centroids_count)

for control_flag in range(0, loop_limit, 1):
    print("Clustering, Loop " + str(control_flag + 1))
    centroid_is_changed = False
    for data in data_set:
        distances = []
        for centroid in centroids:
            distances.append(calculate_distance(data, centroid))
        min_distance = min(distances)

        for index in range(0, len(centroids), 1):
            if distances[index] == min_distance:
                data[2] = index

    for index in range(0, len(centroids), 1):
        coordinates = []
        for data in data_set:
            if index == int(data[2]):
                coordinates.append(data)
        x, y = 0.0, 0.0
        for coordinate in coordinates:
            x += float(coordinate[0])
            y += float(coordinate[1])
        new_centroids = [x / len(coordinates), y / len(coordinates)]
        if new_centroids[0] == centroids[index][0] and new_centroids[1] == centroids[index][1]:
            continue
        centroids[index] = new_centroids
        centroid_is_changed = True
    if not centroid_is_changed:
        print("Centroid converged! Loop stopped.")
        break

result = [list() for i in range(centroids_count)]
for data in data_set:
    result[int(data[2])].append(data.tolist())
for i in range(centroids_count):
    plt.scatter(np.array(result[i])[:, 0], np.array(result[i])[:, 1], s=20)
plt.scatter(np.array(centroids)[:, 0], np.array(centroids)[:, 1], marker='x', s=45)
locator = MultipleLocator(5)
gca = plt.gca()
gca.xaxis.set_major_locator(locator)
gca.yaxis.set_major_locator(locator)
plt.xlim(0, 100)
plt.ylim(0, 100)
legends = []
for i in range(centroids_count):
    legends.append("Cluster " + str(i + 1))
legends.append("Centroids")
plt.legend(legends)
plt.grid()
plt.savefig("./out/kmeans_result.png")
# plt.show()
plt.close()
