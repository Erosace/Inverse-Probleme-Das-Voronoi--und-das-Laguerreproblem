import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
import pandas as pd
import csv


def check_tessellation(data):
    """
    Checks if tessellation is not degenerated

    :param data: data of tessellation, counts the number of edges in a vertex, must be smaller than 4 to be degenerated
    :return: returns True if tessellation data is degenerated
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


def change_data(data):
    """
    check data for double edges and delete one

    :param data: data is a np array with edges inside, e.g data = [[1,2,0.1,0.2,3,4],[2,
    3,0.1,0,4,1]] where the first 2 entries are the labels of the neighbour cells and the other 4 are the coordinates of
     the shared edge
    :return: returns the data without double entries
    """

    data = np.unique(data, axis=0)
    return data


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


def rotating_vector(vector, angle):
    """
    rotates a vector around a certain angle counterclockwise (angle is in rad)

    :param vector: list = [x,y]
    :param angle: float representing the angle in radians
    :return: returns the rotated vector
    """

    # Rotation matrix
    A = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return A @ vector


# line: [Point: [x1,x2], Direction: [y1,y2]]
def intersection_of_two_lines(line1, line2):
    """
    calculates the intersection point of 2 lines in 2D with basic linear algebra

    :param line1: [Point: [x1,y1], Direction: [a1,b1]]
    :param line2: [Point: [x2,y2], Direction: [a2,b2]]
    :return: returns the point as a vector = [x,y]
    """

    # IMPORTANT: It is possible, that a tessellation is not a Voronoi tessellation especially when using the
    # approximation algorithm. Then the two lines given in this function might be parallel either to each other
    # or in some variation. If such thing happens in a variation, lower the variation until the problem does not arise
    # anymore. In the other case a more suitable solution must be found, e.g. not using such lines.
    x1, x2, y1, y2 = line1[0][0], line1[0][1], line2[0][0], line2[0][1]
    a, b, c, d = line1[1][0], line1[1][1], line2[1][0], line2[1][1]
    vector = np.array([x1 - y1, x2 - y2])
    matrix = np.array([[-a, c], [-b, d]])
    x = np.linalg.solve(matrix, vector)
    x = x.tolist()
    arr = [0, 0]
    arr[0], arr[1] = line1[0][0] + x[0] * line1[1][0], line1[0][1] + x[0] * line1[1][1]
    return arr


def calculate_generator_ray(edgeMain1, edgeMain2, edge3):
    """
    Calculate the generating ray for a cell, edgemain1 and edgemain2 are the edges of the main cell adjacent to the
    neighbour cells

    :param edgeMain1: edge = [point1,point2]
    :param edgeMain2: edge = [point3,point4]
    :param edge3: edge = [point5,point6]
    :return: returns a ray: [Point: [x1,y1], Direction: [x2,y2]]
    """

    # sorting the edges such that the first point is the same for every edge
    if edgeMain1[0] in edgeMain2:
        edgeMain2.remove(edgeMain1[0])
        edgeMain2 = np.array([[edgeMain1[0][0], edgeMain1[0][1]], [edgeMain2[0][0], edgeMain2[0][1]]])
        edge3.remove(edgeMain1[0])
        edge3 = np.array([[edgeMain1[0][0], edgeMain1[0][1]], [edge3[0][0], edge3[0][1]]])
    else:
        edgeMain1[0], edgeMain1[1] = edgeMain1[1], edgeMain1[0]
        edgeMain2.remove(edgeMain1[0])
        edgeMain2 = np.array([[edgeMain1[0][0], edgeMain1[0][1]], [edgeMain2[0][0], edgeMain2[0][1]]])
        edge3.remove(edgeMain1[0])
        edge3 = np.array([[edgeMain1[0][0], edgeMain1[0][1]], [edge3[0][0], edge3[0][1]]])

    edgeMain1 = np.array(edgeMain1)
    ray = edge3[0] - edge3[1]
    edge = edgeMain1[1] - edgeMain1[0]
    angle = angle_between_vectors(edge, ray)

    # Check in wich direction the angle has to be applied by the fact that the ray has to be rotated in the inside of
    # the voronoi cell
    if angle_between_vectors(edge,
                             rotating_vector(edgeMain2[1] - edgeMain2[0], np.pi * 2 - angle)) >= angle_between_vectors(
        edge, rotating_vector(edgeMain2[1] - edgeMain2[0], angle)):
        generatingRay = rotating_vector(edgeMain2[1] - edgeMain2[0], angle)
    else:
        generatingRay = rotating_vector(edgeMain2[1] - edgeMain2[0], np.pi * 2 - angle)
    return np.array([edgeMain1[0], generatingRay])


def check_adjacent_cells(cellindex1, cellindex2, adjacentCells):
    """
    checks if 2 cells are adjacent to each other, adjacentCells data is required

    :param cellindex1: cell index of first cell
    :param cellindex2: cell index of second cell
    :param adjacentCells: a list where the ith element is a list of all adjacent cells
    :return: returns True if the 2 cells are adjacent, otherwise False
    """

    for i in range(len(adjacentCells[cellindex1])):
        if adjacentCells[cellindex1][i][1] == cellindex2:
            return True
    return False


def adjacent_adjacent_cells(mainCell, adjacentCells):
    """
    returns a List of lists with the adjacent adjacent cellindizes of a given cell and their indizes

    :param mainCell: the cell we are trying to calculate a generator for
    :param adjacentCells: a list where the ith element is a list of all adjacent cells
    :return: returns a list where each entry is a list of cells that are adjacent to the main cell and each other, also
    returns the cell indizes of the adjacent cells
    """

    cellIndizes = []
    for i in range(len(adjacentCells[mainCell])):
        a = adjacentCells[mainCell][i][1]
        for j in range(i, len(adjacentCells[mainCell])):
            if check_adjacent_cells(a, adjacentCells[mainCell][j][1], adjacentCells):
                cellIndizes += [[a, adjacentCells[mainCell][j][1], i, j]]
    return cellIndizes


def generator_variation(ray1, ray2):
    """
    calculates the approximated generators with their variation

    :param ray1: ray = [Point: [x1,y1], Direction: [x2,y2]]
    :param ray2: ray = [Point: [x3,y3], Direction: [x4,y4]]
    :return: returns their generators and their variation
    """

    # angles to rotate in radiant
    angles = [0.1*i for i in range(-10, 10)]

    generatorWithNoApproximation = intersection_of_two_lines(ray1, ray2)
    approximatedGeneratorsAndVariations = [generatorWithNoApproximation, 0]
    for i in angles:
        for j in angles:
            newRay1 = np.array([ray1[0], rotating_vector(ray1[1], i)])
            newRay2 = np.array([ray2[0], rotating_vector(ray2[1], j)])
            intersection = intersection_of_two_lines(newRay1, newRay2)
            difference = np.linalg.norm(np.array([generatorWithNoApproximation[0]-intersection[0],
                                                  generatorWithNoApproximation[1]-intersection[1]]))
            approximatedGeneratorsAndVariations[1] += difference
    return approximatedGeneratorsAndVariations


def generating_points(data):
    """
    compute the generator points of a given data set

    :param data: data is a np array with edges inside, e.g data = [[1,2,0.1,0.2,3,4],[2,
    3,0.1,0,4,1]] where the first 2 entries are the labels of the neighbour cells and the other 4 are the coordinates of
    the shared edge
    :return: returns the generators
    """

    # Check tessellation if degenerated
    data = change_data(data)

    if check_tessellation(data):
        raise ValueError("The tessellation is degenerated!")
    # Amount of cells
    firstColumn = data.transpose()[0]
    secondColumn = data.transpose()[1]
    amount = max(firstColumn.max(), secondColumn.max()).astype(int) + 1

    # Array that saves the adjacent cells of each cell
    adjacentCells = [[] for i in range(amount)]
    for i in range(len(data)):
        a = data[i][0].astype(int)
        b = data[i][1].astype(int)
        if [a, b, data[i][2], data[i][3], data[i][4], data[i][5]] not in adjacentCells[a]:
            adjacentCells[a] += [[a, b, data[i][2], data[i][3], data[i][4], data[i][5]]]
        if [b, a, data[i][2], data[i][3], data[i][4], data[i][5]] not in adjacentCells[b]:
            adjacentCells[b] += [[b, a, data[i][2], data[i][3], data[i][4], data[i][5]]]

    generators = []
    # calculate 2 rays
    for i in range(amount):
        neighborCells = adjacent_adjacent_cells(i, adjacentCells)
        cellNumber1 = neighborCells[0][0]
        cellNumber2 = neighborCells[0][1]
        neighbor3 = None

        neighbor1 = adjacentCells[i][neighborCells[0][2]]
        neighbor2 = adjacentCells[i][neighborCells[0][3]]

        for j in range(len(adjacentCells[cellNumber1])):
            if adjacentCells[cellNumber1][j][1] == cellNumber2:
                neighbor3 = adjacentCells[cellNumber1][j]
                break

        # calculate ray1
        ray1 = calculate_generator_ray([[neighbor1[2], neighbor1[3]], [neighbor1[4], neighbor1[5]]],
                                       [[neighbor2[2], neighbor2[3]], [neighbor2[4], neighbor2[5]]],
                                       [[neighbor3[2], neighbor3[3]], [neighbor3[4], neighbor3[5]]])

        # calculate ray2
        cellNumber1 = neighborCells[1][0]
        cellNumber2 = neighborCells[1][1]
        neighbor3 = None

        neighbor1 = adjacentCells[i][neighborCells[1][2]]
        neighbor2 = adjacentCells[i][neighborCells[1][3]]

        for j in range(len(adjacentCells[cellNumber1])):
            if adjacentCells[cellNumber1][j][1] == cellNumber2:
                neighbor3 = adjacentCells[cellNumber1][j]
                break

        ray2 = calculate_generator_ray([[neighbor1[2], neighbor1[3]], [neighbor1[4], neighbor1[5]]],
                                       [[neighbor2[2], neighbor2[3]], [neighbor2[4], neighbor2[5]]],
                                       [[neighbor3[2], neighbor3[3]], [neighbor3[4], neighbor3[5]]])
        generators += [intersection_of_two_lines(ray1, ray2)]
    return generators


# Compute generator set with measurement errors
def approximated_generating_points(data):
    """
    compute the generator points of a given data set where the vertices might not be accurate or it is just assumed to
    be a Voronoi tessellation. Therefor it calculates approximative generators.

    :param data: data is a np array with edges inside, e.g data = [[1,2,0.1,0.2,3,4],[2,
    3,0.1,0,4,1]] where the first 2 entries are the labels of the neighbour cells and the other 4 are the coordinates of
    the shared edge
    :return: returns the generators
    """

    # Check tessellation if degenerated
    data = change_data(data)

    if check_tessellation(data):
        raise ValueError("The tessellation is degenerated!")
    # Amount of cells
    firstColumn = data.transpose()[0]
    secondColumn = data.transpose()[1]
    amount = max(firstColumn.max(), secondColumn.max()).astype(int) + 1

    # Array that saves the adjacent cells of each cell
    adjacentCells = [[] for i in range(amount)]
    for i in range(len(data)):
        a = data[i][0].astype(int)
        b = data[i][1].astype(int)
        if [a, b, data[i][2], data[i][3], data[i][4], data[i][5]] not in adjacentCells[a]:
            adjacentCells[a] += [[a, b, data[i][2], data[i][3], data[i][4], data[i][5]]]
        if [b, a, data[i][2], data[i][3], data[i][4], data[i][5]] not in adjacentCells[b]:
            adjacentCells[b] += [[b, a, data[i][2], data[i][3], data[i][4], data[i][5]]]

    generators = []
    # calculate 2 rays
    for i in range(amount):
        neighborCells = adjacent_adjacent_cells(i, adjacentCells)
        approximatedGeneratorsAndVariations = []
        for x in range(len(neighborCells)):
            for y in range(x + 1, len(neighborCells)):
                cellNumber1 = neighborCells[x][0]
                cellNumber2 = neighborCells[x][1]
                neighbor3 = None

                neighbor1 = adjacentCells[i][neighborCells[x][2]]
                neighbor2 = adjacentCells[i][neighborCells[x][3]]

                for j in range(len(adjacentCells[cellNumber1])):
                    if adjacentCells[cellNumber1][j][1] == cellNumber2:
                        neighbor3 = adjacentCells[cellNumber1][j]
                        break

                # calculate ray1
                ray1 = calculate_generator_ray([[neighbor1[2], neighbor1[3]], [neighbor1[4], neighbor1[5]]],
                                               [[neighbor2[2], neighbor2[3]], [neighbor2[4], neighbor2[5]]],
                                               [[neighbor3[2], neighbor3[3]], [neighbor3[4], neighbor3[5]]])

                # calculate ray2
                cellNumber1 = neighborCells[y][0]
                cellNumber2 = neighborCells[y][1]
                neighbor3 = None

                neighbor1 = adjacentCells[i][neighborCells[y][2]]
                neighbor2 = adjacentCells[i][neighborCells[y][3]]

                for j in range(len(adjacentCells[cellNumber1])):
                    if adjacentCells[cellNumber1][j][1] == cellNumber2:
                        neighbor3 = adjacentCells[cellNumber1][j]
                        break

                ray2 = calculate_generator_ray([[neighbor1[2], neighbor1[3]], [neighbor1[4], neighbor1[5]]],
                                               [[neighbor2[2], neighbor2[3]], [neighbor2[4], neighbor2[5]]],
                                               [[neighbor3[2], neighbor3[3]], [neighbor3[4], neighbor3[5]]])
                approximatedGeneratorsAndVariations += [generator_variation(ray1, ray2)]
        overallWeight = 0
        for x in range(len(approximatedGeneratorsAndVariations)):
            overallWeight += 1/approximatedGeneratorsAndVariations[x][1]
        generatorPoints = []
        for x in range(len(approximatedGeneratorsAndVariations)):
            generatorPoints += [[approximatedGeneratorsAndVariations[x][0][0] * (1/(approximatedGeneratorsAndVariations[x][1]*overallWeight)),
                                approximatedGeneratorsAndVariations[x][0][1] * (1/(approximatedGeneratorsAndVariations[x][1]*overallWeight))]]

        generators += [np.sum(np.array(generatorPoints), axis=0)]
    return generators


def write_in_file(generators, filename):
    """
    Write the generators in a given file

    :param generators: The calculated generators of a tessellation
    :param filename:  The filename where the generators shall be saved
    :return: None
    """

    file = open(filename, 'w')
    writer = csv.writer(file)

    writer.writerow(['generator(x)', 'generator(y)', 'radius(r)'])
    for i in range(len(generators)):
        writer.writerow([generators[i][0], generators[i][1], 0])
    file.close()
    return None


# Calculate generators and plot them with the given tessellation
def main(data, filename, approximation=False):
    """
    The main method of inverting Voronoi tessellations.

    :param data: tessellation data as a pandas dataframe
    :param filename: the name of the file where the calculated generators shall be saved
    :param approximation: True if you want to use the approximation algorithm (recommended if you are not sure or
    certainly know that the tessellation is not a Voronoi tessellation) or False if you are sure that the
    tessellation is a Voronoi tessellation.
    :return:
    """
    if not approximation:
        generators = np.array(generating_points(data))
        write_in_file(generators, filename)
    elif approximation:
        generators = np.array(approximated_generating_points(data))
        write_in_file(generators, filename)
    else:
        print("The parameter approximation must be True or False!")

    '''
    voronoi = scipy.spatial.Voronoi(generators)
    fig = scipy.spatial.voronoi_plot_2d(voronoi)
    plt.show()
    '''
    return None


if __name__ == "__main__":
    # Tessellation data as in paper, but the labeling of the cells starts with 0 rather than 1

    tessellation = pd.read_csv("voronoi_tessellation_data - appr.csv")
    # tessellation = pd.read_csv("laguerre_tessellation_data.csv")
    main(tessellation, filename='calculated_voronoi_generators_data.csv')
    main(tessellation, filename='calculated_voronoi_generators_data - Kopie.csv', approximation=True)
