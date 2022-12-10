import numpy as np
import pypoman as pp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from scipy.spatial import ConvexHull
import csv


def change_data(data):
    """
    check data for double generators and delete one, the function also ensures that
    every radius is positiv
    :param data: data is a np array with generators inside, e.g data = [[1,1,1],[2,
    3,3]] where the first 2 entries are the coordinates and the third one is the radius,
    :return: returns the data with only positive radii and without double generators
    """

    data = np.unique(data, axis=0)
    minimum = 0
    for i in range(len(data)):
        if data[i][2] < minimum:
            minimum = data[i][2]
    if minimum < 0:
        for i in range(len(data)):
            data[i][2] -= minimum
    return data


def halfspace(generator1, generator2):
    """
    :param generator1: generator1 = [x1,y1,r1]
    :param generator2: generator2 = [x2,y2,r2]
    :return: returns the halfspace of 2 generating circles
    """

    # halfspace in form of A*x+B*y = k
    A = 2 * (generator1[0] - generator2[0])
    B = 2 * (generator1[1] - generator2[1])
    k = generator1[0] ** 2 + generator1[1] ** 2 - generator1[2] ** 2 - generator2[0] ** 2 - generator2[1] ** 2 + \
        generator2[2] ** 2
    return A, B, k


def cells_vertices(data):
    """
    uses a system of linear equations that characterise the cells to calculate the vertices

    :param data: needs the generating points
    :return: returns the vertices of each cell
    """

    cells = [[] for i in range(len(data))]

    # for plotting purposes
    minx, miny, maxx, maxy, maxr = 0, 0, 0, 0, 0

    for i in range(len(data)):
        if data[i][0] < minx:
            minx = data[i][0]
        if data[i][0] > maxx:
            maxx = data[i][0]
        if data[i][1] < miny:
            miny = data[i][1]
        if data[i][1] > maxy:
            maxy = data[i][1]
        if data[i][2] > maxr:
            maxr = data[i][2]
    # outer Border can be adjusted, len(data)*maxr + 6 can be changed, depending on the case
    outerBorder = max(3, len(data) * maxr + 6)
    for i in range(len(data)):
        # The diagram should be in a certain rectangle
        A = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        b = [maxx + outerBorder, maxy + outerBorder, -minx + outerBorder, -miny + outerBorder]
        for j in range(len(data)):
            if j != i:
                x, y, z = halfspace(data[i], data[j])
                A += [[-x, -y]]
                b += [-z]
        vertices = pp.compute_polytope_vertices(np.array(A), np.array(b))
        cells[i] = vertices
    return cells, maxx + outerBorder, maxy + outerBorder, minx - outerBorder, miny - outerBorder


def create_tessellation(data, filename="0", block=False, center=False, setTitle=None):
    """
    main function to calculate the vertices of the tessellation, plot the tessellation and write the vertices to a file

    :param data: needs the generating points as a pandas data frame, the data should be around (0,0)
    :param filename: the name of the csv file where the tessellation data will be stored, must be a string, for example
    'laguerre_tessellation_data.csv', optional
    :param block: blocks the plot to analyse it, if you want to show multiple plots, only block the last one, optional
    :param center: center the plot more suitable. Try with True/False what suits you more, optional
    :param setTitle: set the title of the plot, must be a string, optional
    :return: returns None, but tessellation is plotted, vertices are stored and non generating points are printed
    """

    data = data.to_numpy()
    data = change_data(data)

    cells, maxx, maxy, minx, miny = cells_vertices(data)
    cellEdges = [[] for i in range(len(cells))]

    # Catch the case that a generator does not create a cell and print all non generating points in the end
    noCell = []
    for i in range(len(cells) - 1, -1, -1):
        if len(cells[i]) == 0:
            noCell += [data[i]]
            del cells[i]
    for i in range(len(cells)):
        conv = ConvexHull(cells[i])
        neighbourVertices = conv.vertices
        for j in range(len(neighbourVertices)):
            edge = [cells[i][neighbourVertices[j]], cells[i][neighbourVertices[(j + 1) % len(neighbourVertices)]]]
            cellEdges[i] += [edge]

    # creating a plot
    fig, ax = plt.subplots()
    fig = []
    for i in range(len(cellEdges)):  # ith cell
        xCoordinates = []
        yCoordinates = []
        for j in range(len(cellEdges[i])):  # jth edge
            xCoordinates += [cellEdges[i][j][0][0], cellEdges[i][j][1][0]]
            yCoordinates += [cellEdges[i][j][0][1], cellEdges[i][j][1][1]]
            fig += [plt.plot(xCoordinates, yCoordinates, linewidth='0.5', color="#261E1E", linestyle='solid')]

    # draw the outlining lines with the background color
    fig += [plt.plot([minx, maxx], [miny, miny], linewidth='3', color="#FFFFFF")]
    fig += [plt.plot([minx, maxx], [maxy, maxy], linewidth='3', color="#FFFFFF")]
    fig += [plt.plot([minx, minx], [miny, maxy], linewidth='3', color="#FFFFFF")]
    fig += [plt.plot([maxx, maxx], [miny, maxy], linewidth='3', color="#FFFFFF")]

    pointsX, pointsY, pointsR = [], [], []
    for i in range(len(data)):
        pointsX += [data[i][0]]
        pointsY += [data[i][1]]
        pointsR += [data[i][2]]
    circles = []
    # paint = ["green", "yellow", "#8080f0", "purple", "red", "orange", "brown"]
    paint = ['#F2C600']
    for i in range(len(data)):
        circles += [plt.Circle((pointsX[i], pointsY[i]), pointsR[i], color=paint[i % len(paint)], alpha=1, fill=False)]
        ax.add_patch(circles[i])

    # print the generators that do not create a cell necessary
    print(noCell)

    # plot everything

    ax.set_aspect('equal', 'box')
    plt.scatter(pointsX, pointsY, s=7, color='black', marker='.')

    # Change the x- and y-limits of the plot axes to have a better view on the tessellation. Can be commented out to
    # see the plot from far away, in order to set the limits correct the tessellation should be around (0,0)
    xlim, ylim = [0, 0], [0, 0]

    if center:
        for i in range(len(cellEdges)):
            for j in range(len(cellEdges[i])):
                for k in range(len(cellEdges[i][j])):
                    if cellEdges[i][j][k][0] > xlim[1] and np.abs(cellEdges[i][j][k][0] - maxx) > 3:
                        xlim[1] = cellEdges[i][j][k][0] + 2
                    if cellEdges[i][j][k][0] < xlim[0] and np.abs(cellEdges[i][j][k][0] - minx) > 3:
                        xlim[0] = cellEdges[i][j][k][0] - 2
                    if cellEdges[i][j][k][1] > ylim[1] and np.abs(cellEdges[i][j][k][1] - maxy) > 3:
                        ylim[1] = cellEdges[i][j][k][1] + 2
                    if cellEdges[i][j][k][1] < ylim[0] and np.abs(cellEdges[i][j][k][1] - miny) > 3:
                        ylim[0] = cellEdges[i][j][k][1] - 2
        for i in range(len(pointsX)):
            if pointsX[i] + pointsR[i] + 2 > xlim[1]:
                xlim[1] = pointsX[i] + pointsR[i] + 2
            if pointsX[i] - pointsR[i] - 2 < xlim[0]:
                xlim[0] = pointsX[i] - pointsR[i] - 2
            if pointsY[i] + pointsR[i] + 2 > ylim[1]:
                ylim[1] = pointsY[i] + pointsR[i] + 2
            if pointsY[i] - pointsR[i] - 2 < ylim[0]:
                ylim[0] = pointsY[i] - pointsR[i] - 2
        limits = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]  # set the limits equal that the plot is quadratic
        ax.set_xlim(limits)
        ax.set_ylim(limits)
    # set the title of the plot
    if setTitle is not None:
        ax.set_title(setTitle)
    plt.ion()
    plt.show(block=block)

    # write the tessellation data in a file, if NOT degenerated
    if filename != "0":
        write_in_file(cellEdges, filename)
    return None


def display_given_tessellation(data, block=False):
    """
    Plot a given tessellation in order to get some insight how the cells look like and which coordinates could be chosen.
    The cells are also labeled.

    :param data: The tessellation data in a suitable format, needs to be pandas dataframe
    :param block: blocks the plot to analyse it, if you want to show multiple plots, only block the last one, optional
    :return: None, plots the tessellation
    """

    data = data.to_numpy()

    # create a plot
    fig, ax = plt.subplots()
    fig = []

    # determine the number of cells
    amountCells = 1
    for i in range(len(data)):
        if data[i][0] >= amountCells or data[i][1] >= amountCells:
            amountCells = max(data[i][0], data[i][1]) + 1

    # amountCells must be an int
    helper = np.array(amountCells)
    helper = helper.astype(np.int64)
    amountCells = helper

    notAssigned = [i for i in range(amountCells)]

    # for labeling the cells: compute the convex hull of the outlining points of a cell, get the inequalities from the
    # convex hull class, use the inequalities to calculate the chebyshev center of the polytope

    for i in range(len(data)):
        cellIndex = int(data[i][0])
        xCoordinates = [data[i][2], data[i][4]]
        yCoordinates = [data[i][3], data[i][5]]
        fig += [plt.plot(xCoordinates, yCoordinates, linewidth='0.5', color="#261E1E", linestyle='solid')]
        if cellIndex in notAssigned:
            # get all outlining points
            points = []
            for j in range(len(data)):
                if data[j][0] == cellIndex or data[j][1] == cellIndex:
                    points += [[data[j][2], data[j][3]], [data[j][4], data[j][5]]]
            polytope = ConvexHull(points).equations
            A = np.array([[polytope[k][0], polytope[k][1]] for k in range(len(polytope))])
            b = np.array([-polytope[k][2] for k in range(len(polytope))])
            center = pp.polygon.compute_chebyshev_center(A, b)
            text = TextPath((center[0], center[1]), "{}".format(cellIndex), size=1.3)
            plt.gca().add_patch(PathPatch(text, color="black"))
            # ax.text(center[0], center[1], "{}".format(cellIndex), size=6)
            notAssigned.remove(cellIndex)
    # label the reminding cells
    while notAssigned:
        for i in notAssigned:
            points = []
            for j in range(len(data)):
                if data[j][0] == i or data[j][1] == i:
                    points += [[data[j][2], data[j][3]], [data[j][4], data[j][5]]]
            polytope = ConvexHull(points).equations
            A = np.array([[polytope[k][0], polytope[k][1]] for k in range(len(polytope))])
            b = np.array([-polytope[k][2] for k in range(len(polytope))])
            center = pp.polygon.compute_chebyshev_center(A, b)
            text = TextPath((center[0], center[1]), "{}".format(i), size=1.3)
            plt.gca().add_patch(PathPatch(text, color="black"))
            # ax.text(center[0], center[1], "{}".format(cellIndex), size=6)
            notAssigned.remove(i)

    ax.set_aspect('equal', 'box')
    ax.set_title("tessellation")

    plt.show(block=block)


def write_in_file(cellEdges, filename):
    """
    write the vertices of a non degenerated tessellation in a file

    :param cellEdges: needs the edges as tessellation data, len(cellEdges) are the number of cells and len(cellEdges[
    i]) are the number of edges of cell i
    :param filename: name of csv file where the tessellation data will be stored, must be string, for example:
    'laguerre_tessellation_data.csv'
    :return: None the tessellation data will be stored in the given file
    """

    file = open(filename, 'w')
    writer = csv.writer(file)

    # eliminate calculating errors
    cellEdges = eliminating_errors(cellEdges)
    # print data in a file
    tessellationData = []
    for i in range(len(cellEdges)):
        for j in range(len(cellEdges)):
            if j > i:
                for x in range(len(cellEdges[i])):
                    for y in range(len(cellEdges[j])):
                        a = (cellEdges[i][x][0][0] == cellEdges[j][y][0][0] and
                             cellEdges[i][x][0][1] == cellEdges[j][y][0][1] and
                             cellEdges[i][x][1][0] == cellEdges[j][y][1][0] and
                             cellEdges[i][x][1][1] == cellEdges[j][y][1][1])
                        b = (cellEdges[i][x][1][0] == cellEdges[j][y][0][0] and
                             cellEdges[i][x][1][1] == cellEdges[j][y][0][1] and
                             cellEdges[i][x][0][0] == cellEdges[j][y][1][0] and
                             cellEdges[i][x][0][1] == cellEdges[j][y][1][1])
                        if a or b:
                            tessellationData += [[i, j, cellEdges[i][x][0][0], cellEdges[i][x][0][1],
                                                  cellEdges[i][x][1][0], cellEdges[i][x][1][1]]]
    writer.writerow(['cell i', 'cell j', 'vertice1(x)', 'vertice1(y)', 'vertice2(x)', 'vertice2(y)'])
    for i in tessellationData:
        writer.writerow(i)
    file.close()
    return None


def distance(point1, point2):
    """
    calculates the euclidean distance of 2 points in 2-dimensional space

    :param point1: list [x1,y1]
    :param point2: list [x2,y2]
    :return: a value in R
    """

    norm = np.linalg.norm(np.array([point1[0] - point2[0], point1[1] - point2[1]]))
    norm = norm.item()
    return norm


def eliminating_errors(data):
    """
    the calculation of the vertices of each cell can make errors for vertices shared by two neighbour cells
    therefore we change the value of the shared vertices to be the arithmetic mean of them

    :param data: needs tessellation data as in function write_in_file, variable cellEdges
    :return: returns the tessellation data, where the almost same vertices are changed to have the same values
    """

    # calculates an upper bound for the maximal error to be acquired
    minimumDistance = 10
    for i in range(len(data)):
        for j in range(len(data[i])):
            x = data[i][j][0]
            y = data[i][j][1]
            length = distance(x, y) / 10
            if length != 0 and length < minimumDistance:
                minimumDistance = length

    # saves the tessellation data as a flattened array
    allVertices = []
    for i in range(len(data)):
        for x in range(len(data[i])):
            for m in range(len(data[i][x])):
                allVertices += [data[i][x][m]]

    # saves all the indices in an array, can be used to pop all used indices of changed vertices
    numbers = [x for x in range(len(allVertices))]

    # gets all the similar vertices, calculates their mean and changes them in the original tessellation
    for i in numbers:
        similarVertices = [allVertices[i]]
        indices = [i]
        for j in range(len(allVertices)):
            if distance(allVertices[i], allVertices[j]) <= minimumDistance:
                similarVertices += [allVertices[j]]
                indices += [j]
        x = 0
        y = 0
        for a in similarVertices:
            x += a[0]
            y += a[1]
        x = x / len(similarVertices)
        y = y / len(similarVertices)
        usedVertices = [allVertices[o] for o in indices]
        for a in range(len(data)):
            for b in range(len(data[a])):
                for c in range(len(data[a][b])):
                    for d in range(len(usedVertices)):
                        if data[a][b][c][0] == usedVertices[d][0] and data[a][b][c][1] == \
                                usedVertices[d][1]:
                            data[a][b][c] = [x, y]
    return data


if __name__ == "__main__":
    '''
    Voronoi Case
    '''
    '''
    generators1 = pd.read_csv("voronoi_generators_data.csv")
    create_tessellation(generators1, filename='voronoi_tessellation_data.csv', center=False, setTitle="original tessellation")
    generators2 = pd.read_csv("calculated_voronoi_generators_data.csv")
    create_tessellation(generators2, block=True, center=False, setTitle="tessellation of calculated generators")
    '''
    ''' 
    Laguerre Case
    '''
    display_given_tessellation(pd.read_csv("laguerre_tessellation_data.csv"), block=True)
    # generators2 = pd.read_csv("calculated_laguerre_generators_data.csv")
    # generators1 = pd.read_csv("laguerre_generators_data.csv")
    # create_tessellation(generators2, filename='laguerre_tessellation_data.csv', block=True, center=False)
    # generators1 = pd.read_csv("voronoi_generators_data.csv")
    # generators2 = pd.read_csv("calculated_voronoi_generators_data.csv")
    # generators3 = pd.read_csv("calculated_voronoi_generators_data - Kopie.csv")
    # create_tessellation(generators1, block=False, center=True, setTitle="original", filename="voronoi_tessellation_data.csv")
    # create_tessellation(generators2, block=False, center=True, setTitle="normaler Algorithmus")
    # create_tessellation(generators3, block=False, center=True, setTitle="approximativer Algorithmus")
    # generators4 = pd.read_csv("voronoi_tessellation_data - appr.csv")
    # display_given_tessellation(generators4, block=False)
    # generator5 = pd.read_csv("calculated_voronoi_generators_data_appr.csv")
    # create_tessellation(generator5, block=True, setTitle="laguerre")

