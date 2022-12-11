import numpy as np
import pandas as pd
import csv
from operator import itemgetter


def distance(point1, point2):
    """
    Using the euclidean norm in np.linalg to calculate the euclidean distance between 2 points

    :param point1: array: [x1, y1, z1, ...]
    :param point2: array: [x2, y2, z2, ...]
    :return: Return the distance between point1 and point2
    """

    norm = np.linalg.norm(point1 - point2)
    norm = norm.item()
    return norm


def check_tessellation(data):
    """
    Check if tessellation is not degenerated, returns True if it is degenerated

    :param data: The tessellation data
    :return: True if data is degenerated, otherwise False
    """

    vertices = [[data[x][y], data[x][y + 1]] for x in range(len(data)) for y in [2, 4]]
    uniqueVertices = np.unique(np.array(vertices), axis=0)
    for i in uniqueVertices:
        count = 0
        for j in range(len(vertices)):
            if vertices[j][0] == i[0] and vertices[j][1] == i[1]:
                count += 1
        if count >= 4:
            return True
    return False


def normalised_vector(vector):
    """
    :param vector: list = [x1,x2]
    :return: returns the vector with the same direction but euclidean length 1
    """

    return vector / np.linalg.norm(vector)


def angle_between_vectors(vector1, vector2):
    """
    Computes the angle between 2 vectors using basic linear algebra

    :param vector1: list = [x1, y1]
    :param vector2: list = [x2, y2]
    :return: returns the angle between the vectors in radians
    """

    v1_norm = normalised_vector(vector1)
    v2_norm = normalised_vector(vector2)
    return np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1, 1))


def check_coordinates(x1, y1, data, x2=None, y2=None):
    """
    Checks if the given coordinates (x2 or y2) is suitable by looking at the angle between the vector of the ray
    perpendicular to the edge of the cell and either the vector [0,1] if x2 is given or [1,0] if y2 is given.

    :param x1: first coordinate of the first generator
    :param y1: second coordinate of the first generator
    :param data: tessellation data (in suitable format)
    :param x2: first (wanted) coordinate of the second generator
    :param y2: second (wanted) coordinate of the second generator
    :return: Returns either False and a message to choose a different coordinate or True with the chosen coordinate
    """

    angle = 0
    edge = which_edge(data, 0, 1)
    ray = perpendicular_line([[x1, y1]], edge)
    vector1 = [ray[1][0], ray[1][1]]
    if y2 is None:
        vector2 = [0, 1]
        angle = min(angle_between_vectors(vector1, vector2), angle_between_vectors([-vector1[0], -vector1[1]], vector2))
        if angle < 0.7:
            return False, "It is unreasonable to choose x2. The angle between the perpendicular ray and your chosen " \
                          "ray is {0} radians. Choose the coordinate y2!".format(angle)
        return True, "x2"
    if x2 is None:
        vector2 = [1, 0]
        angle = min(angle_between_vectors(vector1, vector2), angle_between_vectors([-vector1[0], -vector1[1]], vector2))
        if angle < 0.7:
            return False, "It is unreasonable to choose y2. The angle between the perpendicular ray and your chosen " \
                          "ray is {0} radians. Choose the coordinate x2!".format(angle)
        return True, "y2"
    else:
        vector2 = [0, 1]
        angle1 = min(angle_between_vectors(vector1, vector2),
                     angle_between_vectors([-vector1[0], -vector1[1]], vector2))
        vector3 = [1, 0]
        angle2 = min(angle_between_vectors(vector1, vector3),
                     angle_between_vectors([-vector1[0], -vector1[1]], vector3))
        if angle1 >= angle2:
            return True, "x2"
        else:
            return True, "y2"


def change_data(data):
    """
    Delete multiple entry's in the tessellation data and assure that cell labels are integers.

    :param data: tessellation data
    :return: adjusted data
    """

    data = np.unique(data, axis=0)
    for i in range(len(data)):
        data[i][0], data[i][1] = int(data[i][0]), int(data[i][1])
    return data


def intersection_of_two_lines(line1, line2):
    """
    Calculates the intersection point of 2 lines.

    :param line1: [Point1: [x1, y1], Direction1: [x1', y1']]
    :param line2:  [Point2: [x2, y2], Direction2: [x2', y2']]
    :return: Returns the intersection point of the 2 rays
    """

    x1, x2, y1, y2 = line1[0][0], line1[0][1], line2[0][0], line2[0][1]
    a, b, c, d = line1[1][0], line1[1][1], line2[1][0], line2[1][1]
    vector = np.array([x1 - y1, x2 - y2])
    matrix = np.array([[-a, c], [-b, d]])
    x = np.linalg.solve(matrix, vector)
    x = x.tolist()
    arr = [0, 0]
    arr[0], arr[1] = line1[0][0] + x[0] * line1[1][0], line1[0][1] + x[0] * line1[1][1]
    return arr


def perpendicular_line(point, edge):
    """
    Compute the perpendicular line from one generator point to an edge.

    :param point: Point1: [x1, y1]
    :param edge: [Point2: [x2, y2], Point3: [x3, y3]]
    :return: returns a ray: [Point4: [x4, y4], Direction: [x5, y5]]
    """

    line = np.array([[edge[0], edge[1]], [edge[2] - edge[0], edge[3] - edge[1]]])
    direction = np.cross(np.array([line[1][0], line[1][1], 0]), np.array([0, 0, 1]))
    direction = direction.tolist()
    return [point[0], [direction[0], direction[1]]]


def weight_of_generator(generator1, point2, edge1_3):
    """
    Calculate the weight of a generator point. In order to get the radius of a generator point, you have to subtract the
    minimum weight from all generator points so that the weight of each point is at least 0. Then taking the square
    root gives the radius of each generator point.

    :param generator1: [Point1: [x1, y1], weight: w1, cellIndex1: c1]
    :param point2: Point2: [x2, y2]
    :param edge1_3: [Point 3: [x3, y3], Point4: [x4, y4]]
    :return: returns the weight of point2
    """

    q = np.array([edge1_3[0], edge1_3[1]])
    r = distance(generator1[0], q) ** 2 - generator1[1] - distance(point2, q) ** 2
    return -r


# calculates the next generator dependent on the generators and edges of 2 adjacent cells
def next_generator(generator1, generator2, edge1_3, edge2_3):
    """
    Calculate the next generator with 2 adjacent generators and its adjacent cell edges.

    :param generator1: [Point1: [x1, y1], weight1: w2, cellIndex1: c1]
    :param generator2: [Point2: [x2, y2], weight2: w2, cellIndex2: c2]
    :param edge1_3: [Point3: [x3, y3], Point4: [x4, y4]] The edge between the cell generated by generator1 and the
    generator to be calculated
    :param edge2_3: [Point5: [x5, y5], Point6: [x6, y6]] The edge between the cell generated by generator2 and the
    generator to be calculated
    :return: returns the generator and its weight of cell with edges edge1_3 and edge1_2
    """

    line1 = perpendicular_line(generator1, edge1_3)
    line2 = perpendicular_line(generator2, edge2_3)
    point3 = intersection_of_two_lines(line1, line2)
    w = weight_of_generator(generator1, point3, edge1_3)
    return point3, w


def second_generator(generator1, edge1_2, x2, y2, coordinate):
    """
    Calculates the second generator.

    :param generator1: [Point1: [x1, y1], weight1: w2, cellIndex1: c1]
    :param edge1_2: [Point2: [x2, y2], Point3: [x3, y3]]
    :param x2: The x-coordinate of the second generator given by the user or None.
    :param y2: The y-coordinate of the second generator given by the user or None.
    :param coordinate: Either "x2" or "y2" depending on which coordinate is more suitable for the second generator.
    :return: returns the second generator [Point3: [x3, y3], weight2: w2, cellIndex2: c2]
    """

    if coordinate == "x2":
        line1 = perpendicular_line(generator1, edge1_2)
        point2 = intersection_of_two_lines(line1, np.array([[x2, 0], [0, 1]]))
        return [point2, weight_of_generator(generator1, point2, edge1_2), 1]
    if coordinate == "y2":
        line1 = perpendicular_line(generator1, edge1_2)
        point2 = intersection_of_two_lines(line1, np.array([[0, y2], [1, 0]]))
        return [point2, weight_of_generator(generator1, point2, edge1_2), 1]
    else:
        print("Something went wrong in the function second_generator() or before!")
        return None


def which_edge(data, adjacentcell1, adjacentcell2):
    """
    Determine the edge between 2 cells.

    :param data: tessellation data
    :param adjacentcell1: cell index of first cell
    :param adjacentcell2: cell index of adjacent cell of first cell
    :return: returns the edge between the adjacent cells
    """

    small = min(adjacentcell1, adjacentcell2)
    big = max(adjacentcell1, adjacentcell2)
    for i in range(len(data)):
        if data[i][0] == small and data[i][1] == big:
            return [data[i][2], data[i][3], data[i][4], data[i][5]]


def adjacent_cells(data, cell):
    """
    Determine a list of all adjacent cells.

    :param data: tessellation data
    :param cell: cell index
    :return: returns a list of cell indizes that are adjacent to the given cell
    """

    adjacent = []
    for i in range(len(data)):
        if data[i][0] == cell:
            if data[i][1] not in adjacent:
                adjacent.append(data[i][1])
        if data[i][1] == cell:
            if data[i][0] not in adjacent:
                adjacent.append(data[i][0])
    return adjacent


def generating_points(x1, y1, w1, data, x2=None, y2=None):
    """
    Give the coordinates of the first generator of the first cell and x2 or y2 of the second generator. The coordinates
    of the first generator must be in its generating cell, the coordinates of the second generator must be given as
    follows:
    - give x2 coordinate if the first cell is left/right to the second cell
    - give y2 coordinate if the first cell is above/under the second cell
    It is also possible to choose x2 and y2. The algorithm selects one depending on if the cell is right/left or
    below/above.
    Choose x2 and y2 such that the second generator really lies towards the second cell and not further away than x1,y1.

    :param x1: x-coordinate of generator 1
    :param y1: y-coordinate of generator 1
    :param w1: weight of generator 1, at the end of the calculation the weight is adjusted to be the radius (the lowest
    weight will be subtracted from all weights and then the radius of a generator is equal to the square root of its
    weight)
    :param x2: x-coordinate of generator 2
    :param y2: y-coordinate of generator 2
    :param data: tessellation data as pandas dataframe
    :return: returns the calculated generating points
    """

    data = change_data(data)

    # check tessellation if not degenerated, also checks if x1,y1 lies in the first cell, x2/y2 is chosen suitably
    if check_tessellation(data):
        raise ValueError("The tessellation is degenerated!")
    if x2 is None and y2 is None:
        raise ValueError("Choose either x2 or y2 according to the postion of the first 2 cells. Neither x2 nor y2 was "
                         "given. See documentation of generating_points() for more help.")
    if not generator_in_cell([[x1, y1], w1, 0], data):
        raise ValueError("The first generator does not seem to be in its generated cell!")

    # the first coordinate does not have to be in its generated cell, but the relationship between the first 2 points
    # have to be in a certain way
    # coordinate is either a message for the user if right is False or the coordinate chosen
    right, coordinate = check_coordinates(x1, y1, data, x2=x2, y2=y2)
    if not right:
        raise ValueError(coordinate)

    # determine the number of cells
    amountCells = 1
    for i in range(len(data)):
        if data[i][0] > amountCells or data[i][1] > amountCells:
            amountCells = max(data[i][0], data[i][1]) + 1

    # amountCells must be an int
    help = np.array(amountCells)
    help = help.astype(np.int64)
    amountCells = help

    # p stores the generating points, [coordinates, weight, cell_number]
    p = [[[x1, y1], w1, 0]]
    p.append(second_generator(p[0], which_edge(data, 0, 1), x2, y2, coordinate))

    # a list of (not yet) assigned cells
    notAssigned = [x for x in range(2, amountCells)]
    assigned = [0, 1]

    # a list of all cells containing their adjacent cells, could probably be done differently
    # information is already in tessellation
    adjacent = []
    for i in range(2, amountCells):
        adjacent.append(adjacent_cells(data, i))

    while notAssigned:
        for i in notAssigned:
            number = 0
            assignedAdjacent = []
            for l in adjacent[i - 2]:
                # check if at least 2 adjacent cells already have their generator
                if l in assigned:
                    number += 1
                    assignedAdjacent.append(l)
                if number >= 2:
                    generator1 = None
                    generator2 = None
                    for k in range(len(p)):
                        if p[k][2] == assignedAdjacent[0]:
                            generator1 = [p[k][0], p[k][1]]
                        if p[k][2] == assignedAdjacent[1]:
                            generator2 = [p[k][0], p[k][1]]
                    edge1_3 = which_edge(data, assignedAdjacent[0], i)
                    edge2_3 = which_edge(data, assignedAdjacent[1], i)
                    next_point = next_generator(generator1, generator2, edge1_3, edge2_3)
                    p.append([next_point[0], next_point[1], i])
                    notAssigned.remove(i)
                    assigned.append(i)
                    break

    # calculate the radius of each generating point
    weight = [p[i][1] for i in range(len(p))]
    if min(weight) < 0:
        k = min(weight)
        for i in range(len(p)):
            p[i][1] -= k
    for i in range(len(p)):
        p[i][1] = np.sqrt(p[i][1]).item()
    return p


def write_in_file(generators, filename):
    """
    Write the calculated generators in a file.

    :param generators: An array of generators. For example: [[[x1, y1], r1, c1], [[x2, y2], r2, c2], ...]
    :param filename: The name of file where the calculated generators will be stored
    :return: None
    """

    file = open(filename, 'w')
    writer = csv.writer(file)

    writer.writerow(['generator(x)', 'generator(y)', 'radius(r)'])
    for i in range(len(generators)):
        writer.writerow([generators[i][0][0], generators[i][0][1], generators[i][1]])
    file.close()
    return None


def generating_first_four_points(x1, y1, w1, data, x2=None, y2=None):
    """
    Nearly the same as the function generating_points() but only the first four points are calculated in order to check
    if the sample for the approximation algorithm is suitable.
    Check generating_points for more details.

    :param x1: x-coordinate of generator 1
    :param y1: y-coordinate of generator 1
    :param w1: weight of generator 1, at the end of the calculation the weight is adjusted to be the radius (the lowest
    weight will be subtracted from all weights and then the radius of a generator is equal to the square root of its
    weight)
    :param x2: x-coordinate of generator 2
    :param y2: y-coordinate of generator 2
    :param data: tessellation data as pandas dataframe
    :return: returns the first four generating points
    """

    # check tessellation if not degenerated, also checks if x2/y2 is chosen suitably
    if check_tessellation(data):
        raise ValueError("The tessellation is degenerated!")

    # the first coordinate does not have to be in its generated cell, but the relationship between the first 2 points
    # have to be in a certain way
    # coordinate is either the message for the user if right is False or the coordinate chosen
    right, coordinate = check_coordinates(x1, y1, data, x2=x2, y2=y2)
    if not right:
        raise ValueError(coordinate)

    # determine the number of cells
    amountCells = 1
    for i in range(len(data)):
        if data[i][0] >= amountCells or data[i][1] >= amountCells:
            amountCells = max(data[i][0], data[i][1]) + 1

    # amountCells must be an int
    helper = np.array(amountCells)
    helper = helper.astype(np.int64)
    amountCells = helper

    # p stores the generating points, [coordinate, weight, cell_number]
    p = [[[x1, y1], w1, 0]]
    p.append(second_generator(p[0], which_edge(data, 0, 1), x2, y2, coordinate))

    # a list of (not yet) assigned cells
    notAssigned = [x for x in range(2, amountCells)]
    assigned = [0, 1]

    # a list of all cells containing their adjacent cells, could probably be done differently
    # information is already in tessellation
    adjacent = []
    for i in range(2, amountCells):
        adjacent.append(adjacent_cells(data, i))

    counter = 0
    for i in notAssigned:
        number = 0
        assignedAdjacent = []

        # stops when the first 4 cells have generators
        if counter >= 4:
            break
        for l in adjacent[i - 2]:
            # check if at least 2 adjacent cells already have their generator
            if l in assigned:
                number += 1
                assignedAdjacent.append(l)
            if number >= 2:
                generator1 = None
                generator2 = None
                for k in range(len(p)):
                    if p[k][2] == assignedAdjacent[0]:
                        generator1 = [p[k][0], p[k][1]]
                    if p[k][2] == assignedAdjacent[1]:
                        generator2 = [p[k][0], p[k][1]]
                edge1_3 = which_edge(data, assignedAdjacent[0], i)
                edge2_3 = which_edge(data, assignedAdjacent[1], i)
                next_point = next_generator(generator1, generator2, edge1_3, edge2_3)
                p.append([next_point[0], next_point[1], i])
                notAssigned.remove(i)
                assigned.append(i)
                counter += 1
                break

    # calculate the radius of each generating point
    weight = [p[i][1] for i in range(len(p))]
    if min(weight) < 0:
        k = min(weight)
        for i in range(len(p)):
            p[i][1] -= k
    for i in range(len(p)):
        p[i][1] = np.sqrt(p[i][1]).item()
    return p


def approximated_generating_points(data, sampleSize=20, rarityParameter=0.2, mu=None, sigma=None, epsilon=None):
    """
    Calculate a set of generating points with a minimal maximum radius greater than 0. At least the first 4 generators
    have to be in their generated cells.
    If the runtime is too large, the parameters of the multivariate normal distribution might not be chosen reasonable.

    :param data: tessellation data as a pandas dataframe
    :param sampleSize: The size of the sample to draw the coordinates x1, y1, x2, y2 from
    :param rarityParameter: The share of the sampleSize that is used for the elite sample
    :param mu: Optional: the mean of the 4 dimensional multi variate normal distribution, must be a numpy array. If you
    chose to not specify mu the algorithm will use a preset mean that might take very long to compute a result.
    :param sigma: Optional: the 4x4 Covariance Matrix of the multi variate normal distribution as a numpy array. If you
    chose to not specify sigma the algorithm will use a preset covariance matrix that might take very long to compute
    a result.
    :param epsilon: The minimum accuracy so that the algorithm terminates
    :return: returns a set of generating points with minimum maximum radius greater than 0, also prints the amount of
    iterations needed, the deviation and the covariance matrix
    """

    # standard deviation, covariance matrix and mean of multivariate normal distribution if None is given
    if sigma is None:
        sigma = np.array([[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]])
    if mu is None:
        mu = np.array([-4, -1, -2, 0])  # x1, y1, x2, y2
    if epsilon is None:
        epsilon = 0.002

    data = change_data(data)

    amountEliteSet = int(rarityParameter * sampleSize)

    deviation = epsilon
    counter = 0
    generators = [[] for i in range(sampleSize)]
    while deviation >= epsilon and counter < 100:
        samples = np.random.multivariate_normal(mu, sigma, size=[sampleSize])
        generators = [[] for i in range(sampleSize)]
        # first make sure that every sample is acceptable
        for i in range(sampleSize):
            x1, y1, x2, y2 = samples[i][0], samples[i][1], samples[i][2], samples[i][3]
            w1 = 0
            generators[i] = generating_first_four_points(x1, y1, w1, data, x2, y2)
            generators[i] = [generator_in_cell(generators[i][x], data) for x in range(len(generators[i]))]

            zahl = 0
            # If one sample is not accepted, ergo the entry is False, another sample is drawn. It may take a while for
            # every sample to be accepted. In order to keep track in which iterations we are, the "number of draws" is
            # printed modulo 500 and the sample we are currently looking at.
            while not check_boolean_list(generators[i]):
                if zahl % 500 == 0:
                    print("amount of iterations:", counter, "number of draws:", zahl, "sample:", i)
                samples[i] = np.random.multivariate_normal(mu, sigma, size=[1])
                x1, y1, x2, y2 = samples[i][0], samples[i][1], samples[i][2], samples[i][3]
                w1 = 0
                generators[i] = generating_first_four_points(x1, y1, w1, data, x2, y2)
                generators[i] = [generator_in_cell(generators[i][x], data) for x in range(len(generators[i]))]
                zahl += 1
        # calculate the whole generating data set
        for i in range(sampleSize):
            x1, y1, x2, y2 = samples[i][0], samples[i][1], samples[i][2], samples[i][3]
            w1 = 0
            generators[i] = generating_points(x1, y1, w1, data, x2, y2)
        radiis = [[get_largest_radii(generators[x]), x] for x in range(len(generators))]
        radiis = sorted(radiis, key=itemgetter(0))
        maxRadii = radiis[amountEliteSet]
        parametersEliteSet = []
        for i in range(len(radiis)):
            if radiis[i][0] <= maxRadii[0]:
                parametersEliteSet += [[samples[radiis[i][1]][0], samples[radiis[i][1]][1], samples[radiis[i][1]][2],
                                        samples[radiis[i][1]][3]]]
        # Update mu from eliteSample
        mean = np.mean(parametersEliteSet, axis=0)
        # Check the deviation from the last step
        deviation = distance(mean, mu)
        # Update sigma from elite sample
        mu = mean
        sigma = np.cov(np.array(parametersEliteSet).T)
        counter += 1
        print("amount of iterations:", counter)
    print("amount of iterations:", counter, "\n", "deviation:", deviation, "\n", sigma)
    return generators[0]


def get_largest_radii(sample):
    """
    Determines the maximum radii of a sample of generators

    :param sample: one set of generating points
    :return: max radii of the Set
    """

    maxRadii = 0
    for i in range(len(sample)):
        if sample[i][1] > maxRadii:
            maxRadii = sample[i][1]
    return maxRadii


def check_boolean_list(array):
    """
    Checks a 1D array if an entry is False.

    :param array: Some 1D array
    :return: returns False if at least one entry is False, otherwise True.
    """

    for i in range(len(array)):
        if not array[i]:
            return False
    return True


def generator_in_cell(generator, data):
    """
    Determines if a generator lies in its generated cell. The tessellation has to be in a suitable formate.

    :param generator: [point1: [x,y], weight: w, cellIndex; c]
    :param data: The tessellation data in the suitable format: if you go along the edge between 2 cells from the first
    point (for example data[i][2],data[i][3]) to the second point of the edge (data[i][4],data[i][5]) and then to the
    cell left to it, it must have the cell index data[i][0]. The cell right has to have the cell index data[i][1].
    :return: returns True if the generator lies in its cell. Otherwise, False is returned.
    """

    cellIndex = generator[2]
    edges = []
    # store all edges of cell cellIndex in a suitable way, such that you go from first point to the second and then to
    # the left and there is your cell
    for i in range(len(data)):
        if data[i][0] == cellIndex:
            edges += [[data[i][2], data[i][3], data[i][4], data[i][5]]]
    for i in range(len(data)):
        if data[i][1] == cellIndex:
            edges += [[data[i][4], data[i][5], data[i][2], data[i][3]]]
    # now instead of an edge, the halfplane is saved in order to compute if the generator lies on the right side
    # edge[i] = ith halfplane, halfplane = [normal vector, support vector]
    for i in range(len(edges)):
        edges[i] = [[-(edges[i][3] - edges[i][1]), (edges[i][2] - edges[i][0])], [edges[i][0], edges[i][1]]]
    for i in range(len(edges)):
        if np.dot(np.array(edges[i][0]),
                  np.array([generator[0][0] - edges[i][1][0], generator[0][1] - edges[i][1][1]])) >= 0:
            continue
        return False
    return True


def main(data, filename, x1=None, y1=None, w1=None, x2=None, y2=None, approximation=False, sampleSize=None,
         rarityParameter=None, mu=None, sigma=None, epsilon=None):
    """
    The main method of inverting Laguerre tessellations. Uses either the algorithm generating_points() or the algorithm
    approximated_generating_points() to calculate the generators.
    Depending on your choice of algorithms (ergo set approximation=True or approximation=False) some parameters are
    mandatory and others are optional. If you want to make sure that everything runs smoothly give a reasonable value
    for every parameter.
    Check the documentation of the algorithms for more information about the parameters.

    :param data: The tessellation data as a pandas dataframe
    :param filename: The filename where the generators shall be saved
    :param x1: The first coordinate of the generator of the first cell. Mandatory if you do not want to use the
    approximation (ergo approximation=False).
    :param y1: The second coordinate of the generator of the first cell. Mandatory if you do not want to use the
    approximation (ergo approximation=False).
    :param w1: The weight of the generator of the first cell. Mandatory if you do not want to use the
    approximation (ergo approximation=False).
    :param x2: The first coordinate of the generator of the second cell. Either x2 and/or y2 is  mandatory if you do not
    want to use the approximation (ergo approximation=False).
    :param y2: The second coordinate of the generator of the second cell. Either x2 and/or y2 is  mandatory if you do not
    want to use the approximation (ergo approximation=False).
    :param approximation: True or False depending on your decision if you want to use the approximation algorithm for
    the minimal maximum radius or not.
    :param sampleSize: Give the sample size for the approximation algorithm
    :param rarityParameter: Give a rarity parameter in (0,1) for the approximation algorithm
    :param mu: Give the mean of the 4 dimensional multi variate normal distribution. Must be numpy array
    :param sigma: Give the covariance matrix of the 4 dimensional multi variate normal distribution. Must be numpy array
    :param epsilon: Give the minimum accuracy of the approximation algorithm
    :return: None
    """

    if not approximation:
        generators = generating_points(x1, y1, w1, data, x2=x2, y2=y2)
        write_in_file(generators, filename)
    else:
        generators = approximated_generating_points(data, sampleSize, rarityParameter, mu, sigma, epsilon)
        write_in_file(generators, filename)
    return None


if __name__ == '__main__':
    # Tessellation data as in paper, but the labeling of the cells starts with 0 rather than 1
    # Also the tessellation data must be that if you go from the first point of an edge to the second one, the generator
    # of the cell to the left has to have the first label of the data. It is necessary to proof if a generating point
    # lies in its generated cell. Otherwise, a computational more complicated method must be implemented. The
    # tessellation data created by "Calculating Laguerre Cells" already has that property

    tessellation = pd.read_csv("voronoi_tessellation_data - appr.csv")
    # main(tessellation, filename='calculated_laguerre_generators_data.csv',  x1=-10, y1=-10, w1=0.1, x2=-8, y2=1.5)
    main(tessellation, filename='calculated_voronoi_generators_data_appr.csv', approximation=True, sampleSize=50,
         rarityParameter=0.2, mu=[0.1, 2.7, 1.5, 6])
