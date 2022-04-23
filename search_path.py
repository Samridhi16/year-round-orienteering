from PIL import Image
from queue import PriorityQueue
from queue import Queue
import numpy as np
import math
import sys

# input from user using command line arguments
if len(sys.argv) != 6:
    print("Usage:python search_path.py terrain.png mpp.txt red.txt winter redWinter.png")
else:
    image_file = sys.argv[1]
    elevation_file = sys.argv[2]
    events_file = sys.argv[3]
    season = sys.argv[4]
    output_file = sys.argv[5]

# to calculate boundaries around the water
edgelist = []

# priority queue of events node
pq = PriorityQueue()

# to store path calculated by s star
path = []

# read elevation file and convert 500 X 400 into 500 X 395
file = open(elevation_file, 'r')
elevation_data = np.array([line.split() for line in file])
totalrows, totalcolumns = np.shape(elevation_data)
elevation_data = np.delete(elevation_data, np.s_[totalcolumns - 5:totalcolumns + 1], 1)
elevation_data = elevation_data.transpose()
elevation_data = elevation_data.astype('float64')

# read image file
img = Image.open(image_file)
pix = img.load()
terrain1 = img.convert('RGB')

# copy image to output file
outputImage = img.copy()
outputImage1 = outputImage.convert('RGB')

# read events file
file1 = open(events_file, 'r')
events = np.array([line.split() for line in file1])
events = events.astype('int32')

# create lookup table for terrain in summer
terraintype_summer_winter_spring = {(248, 148, 18): 0.6,
                                    (255, 192, 0): 1.1,
                                    (255, 255, 255): 0.8,
                                    (2, 208, 60): 0.6,
                                    (2, 136, 40): 1.5,
                                    (5, 73, 24): -1,
                                    (0, 0, 255): 1.0,
                                    (71, 51, 3): 0.1,
                                    (0, 0, 0): 0.2,
                                    (100, 235, 255): 0.3,
                                    (123, 53, 27): 0.9
                                    }


class Node:
    """
        class node for A star implementation to store attributes of each node
    """
    __slots__ = ['x', 'y', 'z', 'g', 'h', 'f', 'parent']

    def __init__(self, x, y, elevation, parent=()):
        """
        :param x: x coordinate
        :param y: y coordinate
        :param elevation: height
        :param parent: parent node
        """
        self.x = x
        self.y = y
        self.z = elevation
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other_point):
        """
        compares elevations of two points
        :param other_point: second point
        :return: point that has higher elevation
        """
        return self.z < other_point.elevation

    def __eq__(self, other):
        """
        compares nodes if the f values are equal
        :param other: second point
        :return: returns True if both points hav same f values
        """
        return self.x == other.x and self.y == other.y


def add_edges():
    """
    creates a list containing all the edges or boundary pixels of water
    """
    rows = 395
    cols = 500
    for x in range(rows):
        for y in range(cols):
            current_node = x, y
            current_node_color = terrain1.getpixel(current_node)
            if current_node_color[0] == 0 and current_node_color[1] == 0 and current_node_color[2] == 255:
                neighbors = [(x + 1, y), (x - 1, y), (x, y - 1), (x, y + 1)]
                for neighbor in neighbors:
                    x1, y1 = neighbor
                    if rows > x1 >= 0 and cols > y1 >= 0 and neighbor not in edgelist:
                        neighbor_color = terrain1.getpixel(neighbor)
                        if not (neighbor_color[0] == 0 and neighbor_color[1] == 0 and neighbor_color[2] == 255):
                            edgelist.append(neighbor)

    for i in range(len(edgelist)):
        outputImage.putpixel(edgelist[i], (100, 235, 255))


def breath_first_search():
    """
    calculates the mud region or ice region for spring or winter season
    """
    queue = Queue()
    for i in range(len(edgelist)):
        queue.put(edgelist[i])
        while queue.qsize() != 0:
            current_node = queue.get()
            x = current_node[0]
            y = current_node[1]
            if season == "winter":
                for i in range(1, 8):
                    neighbors = [(x + i, y), (x - i, y), (x, y + i), (x, y - i)]
                    for neighbor in neighbors:
                        x1 = neighbor[0]
                        y1 = neighbor[1]
                        if 0 <= x1 < 395 and 0 <= y1 < 500:
                            neighbor_color = terrain1.getpixel(neighbor)
                            if neighbor_color[0] == 0 and neighbor_color[1] == 0 and neighbor_color[2] == 255:
                                outputImage.putpixel(neighbor, (100, 235, 255))
            elif season == "spring":
                for i in range(1, 16):
                    neighbors = [(x + i, y), (x - i, y), (x, y + i), (x, y - i)]
                    for neighbor in neighbors:
                        x1 = neighbor[0]
                        y1 = neighbor[1]
                        if 0 <= x1 < 395 and 0 <= y1 < 500: 
                            neighbor_color = terrain1.getpixel(neighbor)
                            if not (neighbor_color[0] == 0 and neighbor_color[1] == 0 and neighbor_color[2] == 255):
                                if not (neighbor_color[0] == 205 and neighbor_color[1] == 0 and neighbor_color[2] == 101):
                                    if elevation_data[x][y] - elevation_data[x1][y1] < 1:
                                        outputImage.putpixel(neighbor, (123, 53, 27))


def a_star(start_node, end_node):
    """
    A star algorithm to find path from start node to end goal
    :param start_node: source node
    :param end_node: destination node
    :return:
    """
    closed = []
    visited = []

    q = PriorityQueue()
    q.put((0, start_node))
    visited.append(start_node)
    while not q.empty():
        priority, current_node = q.get()

        if current_node in closed:
            continue

        closed.append(current_node)

        # final destination reached
        if current_node == end_node:
            path = []
            while current_node != start_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[:: -1]

        # calculate neighbors
        x = current_node.x
        y = current_node.y

        # create neighbors list
        neighbors = [(x + 1, y), (x - 1, y + 1), (x - 1, y), (x - 1, y - 1), (x, y - 1), (x + 1, y - 1), (x, y + 1),
                     (x + 1, y + 1)]
        for neighbor in neighbors:
            # print(neighbor)
            x, y = neighbor
            if x < 395 and y < 500:
                if season == "summer" or season == "fall":
                    terraincolor = terrain1.getpixel((int(x), int(y)))
                elif season == "winter" or season == "spring":
                    newterrain = outputImage.convert('RGB')
                    terraincolor = newterrain.getpixel((int(x), int(y)))

                # out of bound terrain check
                if terraincolor == (205, 0, 101):
                    continue

                mult_factor = check_terrain_type(terraincolor)

            else:
                break

            if season == "fall":
                neighbors2 = [(x + 1, y), (x - 1, y + 1), (x - 1, y), (x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
                              (x, y + 1), (x + 1, y + 1)]
                for neighbor2 in neighbors2:
                    x_new, y_new = neighbor2
                    neighbor2_color = terrain1.getpixel((int(x_new), int(y_new)))
                    if neighbor2_color[0] == 255 and neighbor2_color[1] == 255 and neighbor2_color[2] == 255:
                        mult_factor = check_terrain_type(terraincolor) * 1.3

            new_entry = Node(neighbor[0], neighbor[1], elevation_data[neighbor[0]][neighbor[1]], current_node)
            new_entry.g = math.sqrt(pow(new_entry.x - start_node.x, 2) + pow(new_entry.y - start_node.y, 2) + pow(
                new_entry.z - start_node.z, 2))
            new_entry.h = math.sqrt(
                pow(new_entry.x - end_node.x, 2) + pow(new_entry.y - end_node.y, 2) + pow(new_entry.z - end_node.z,
                                                                                          2)) * mult_factor
            new_entry.f = new_entry.g + new_entry.h
            sortby = new_entry.f

            if add_in_priority_queue(visited, new_entry):
                visited.append(new_entry)
                q.put((sortby, new_entry))

    return None

def add_in_priority_queue(visited, new_entry):
    """
    adds the elements to priority queue if the node is not already added and the current heuristic is not less than upcoming
    :param visited: list of visited nodes 
    :param new_entry: new node
    """
    for node in visited:
        if new_entry == node and new_entry.f >= node.f:
            return False
    return True

def check_terrain_type(terraincolor):
    """
    to get the heuristic associated with each terrain
    :param terraincolor: color of that terrain
    :return: returns terrain type
    """
    terraincheck = terraintype_summer_winter_spring
    for key in terraincheck:
        if key == terraincolor:
            return terraincheck[key]
    return -1


def create_output_image():
    """
    traverse the path on the image
    :return: distance
    """
    x1 = events[0][0]
    y1 = events[0][1]
    distance = 0
    for point in path:
        x2 = int(point[0])
        y2 = int(point[1])
        outputImage.putpixel((x2, y2), (255, 0, 0))
        distance = distance + abs(x2 - x1) * 10.29 + abs(y2 - y1) * 7.55
        x1 = x2
        y1 = y2
    return distance
 
 
# execution of A star algorithm
if season == "summer" or season == "fall":
    index = 0
    while index < len(events) - 1:
        start_node = Node(events[index][0], events[index][1], elevation_data[events[index][0]][events[index][1]], None)
        end_node = Node(events[index + 1][0], events[index + 1][1],
                        elevation_data[events[index + 1][0]][events[index + 1][1]], None)
        path.extend(a_star(start_node, end_node))
        index += 1
        distance = create_output_image()

if season == "winter" or season == "spring":
    add_edges()
    breath_first_search()
    index = 0
    while index < len(events) - 1:
        start_node = Node(events[index][0], events[index][1], elevation_data[events[index][0]][events[index][1]], None)
        end_node = Node(events[index + 1][0], events[index + 1][1],
                        elevation_data[events[index + 1][0]][events[index + 1][1]], None)
        path.extend(a_star(start_node, end_node))
        index += 1
        distance = create_output_image()

print("Total Distance:", distance)
outputImage.save(output_file)
