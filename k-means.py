import math
import random
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='K-means clustering')

    parser.add_argument('num_clusters',
                        action='store', type=int,
                        help='The number of clusters to required')

    parser.add_argument('input_file', action='store',
                        help='The input file containing data points')

    parser.add_argument('out_file', action='store',
                        help='The output file containing final clusters')

    args = parser.parse_args()
    start(args.num_clusters, 25,
          args.input_file, args.out_file)


def start(num_clusters, max_iterations, input_filename, out_filename):

    opt_cutoff = 0.001

    import csv

    with open(input_filename, 'rU') as f:
        reader = csv.reader(f, delimiter="\t")
        _lst = list(reader)
        _lst.pop(0)
        points = [Point([float(x), float(y)], _id) for _id, x, y in _lst]

    clusters = kmeans(points, num_clusters, opt_cutoff, max_iterations)
    lines = []
    
    with open(out_filename, 'w+') as f:
        for i, c in enumerate(clusters):
            point_ids = []
            for p in c.points:
                point_ids.append(p.id)
            lines.append("Cluster %s : \t %s \n" % (i, ",".join(point_ids)))
        f.writelines(lines)

    print computeSSE(clusters)


class Point:
    
    def __init__(self, coords, _id=None):
        self.id = _id
        self.coords = coords
        self.n = len(coords)

    def __repr__(self):
        return str(self.coords)


class Cluster:
    
    def __init__(self, points):
        if len(points) == 0:
            raise Exception("ILLEGAL: empty cluster")

        # The points that belong to this cluster
        self.points = points

        # The dimensionality of the points in this cluster
        self.n = points[0].n

        # Assert that all points are of the same dimensionality
        for p in points:
            if p.n != self.n:
                raise Exception("ILLEGAL: wrong dimensions")

        # Set up the initial centroid (this is usually based off one point)
        self.centroid = self.calculateCentroid()

    def __repr__(self):
        
        return str(self.points)

    def update(self, points):
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculateCentroid()
        shift = getEucledianDistance(old_centroid, self.centroid)
        return shift

    def calculateCentroid(self):
        numPoints = len(self.points)
        # Get a list of all coordinates in this cluster
        coords = [p.coords for p in self.points]
        # Reformat that so all x's are together, all y'z etc.
        unzipped = zip(*coords)
        # Calculate the mean for each dimension
        centroid_coords = [math.fsum(dList)/numPoints for dList in unzipped]

        return Point(centroid_coords)


def kmeans(points, k, cutoff, max_iterations):

    # Pick out k random points to use as our initial centroids
    initial = random.sample(points, k)

    # Create k clusters using those centroids
    clusters = [Cluster([p]) for p in initial]

    # Loop through the dataset until the clusters stabilize
    loopCounter = 0
    while True:
        # Create a list of lists to hold the points in each cluster
        lists = [[] for c in clusters]
        clusterCount = len(clusters)

        # Start counting loops
        loopCounter += 1
        # For every point in the dataset ...
        for p in points:
            # Get the distance between that point and the centroid of the first
            # cluster.
            smallest_distance = getEucledianDistance(p, clusters[0].centroid)

            # Set the cluster this point belongs to
            clusterIndex = 0

            # For the remainder of the clusters ...
            for i in range(clusterCount - 1):
                # calculate the distance of that point to each other cluster's
                # centroid.
                distance = getEucledianDistance(p, clusters[i+1].centroid)
                # If it's closer to that cluster's centroid update what we
                # think the smallest distance is, and set the point to belong
                # to that cluster
                if distance < smallest_distance:
                    smallest_distance = distance
                    clusterIndex = i+1
            lists[clusterIndex].append(p)

        # Set our biggest_shift to zero for this iteration
        biggest_shift = 0.0

        # As many times as there are clusters ...
        for i in range(clusterCount):
            # Calculate how far the centroid moved in this iteration
            shift = clusters[i].update(lists[i])
            # Keep track of the largest move from all cluster centroid updates
            biggest_shift = max(biggest_shift, shift)

        # If the centroids have stopped moving much, say we're done!
        if biggest_shift < cutoff and loopCounter <= max_iterations:
            print "Clustering iterations run : %s " % loopCounter
            break
    return clusters


def getEucledianDistance(a, b):
    ret = reduce(lambda x, y: x + pow((a.coords[y]-b.coords[y]), 2),
                 range(a.n), 0.0)
    return math.sqrt(ret)


def computeSSE(clusters):
    sse = 0.0
    for c in clusters:
        temp = 0.0
        c_centroid = c.centroid.coords
        for p in c.points:
            temp += getEucledianDistance(p, c.centroid)
        sse += temp
    return sse


if __name__ == "__main__":
    main()
