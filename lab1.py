from PIL import Image
import numpy as np
#import sys

# if len(sys.argv) !=6:
#     print("Usage:python lab1.py terrain.png mpp.txt red.txt winter redWinter.png")
# else:
#     image_file = sys.argv[1]
#     elevation_file = sys.argv[2]
#     events_file = sys.argv[3]
#     season = sys.argv[4]
#     output_file = sys.argv[5]


file = open('mpp.txt', 'r')
elevation_data = []
elevation_data = np.array([line.split() for line in file])
totalrows, totalcolumns = np.shape(elevation_data)
elevation_data = np.delete(elevation_data, np.s_[totalcolumns-5:totalcolumns+1], 1)
print(np.shape(elevation_data))
print(elevation_data)




img = Image.open("terrain.png")
imgheight, imgweight = img.size
for i in range(imgheight):
     for j in range(imgweight):
          coordinate =x,y = i,j
          print(img.getpixel(coordinate))


from PIL import Image
from queue import PriorityQueue
from queue import Queue
import numpy as np
import math
import sys

# season

# priority queue of events node
pq = PriorityQueue()
path = []
# read elevation file and convert 500 X 400 into 500 X 395
file = open('mpp.txt', 'r')

elevation_data = np.array([line.split() for line in file])
totalrows, totalcolumns = np.shape(elevation_data)
elevation_data = np.delete(elevation_data, np.s_[totalcolumns - 5:totalcolumns + 1], 1)
#print(np.shape(elevation_data))
elevation_data = elevation_data.transpose()
#print(np.shape(elevation_data))
elevation_data =elevation_data.astype('float64')

#read image file
img = Image.open('terrain.png')
terrain1 = img.convert('RGB')

#copy image to output file
outputImage = img.copy()
outputImage1 = outputImage.convert('RGB')


# read events file
file1 = open('red.txt', 'r')
events = np.array([line.split() for line in file1])
events = events.astype('int32')


#create lookup table for terrain
terraintype={(248, 148, 18):0.2,
             (255, 192, 0):0.5,
             (255, 255, 255):0.4,
             (2, 208, 60):0.3,
             (2, 136, 40):0.6,
             (5, 73, 24):-1,
             (0, 0, 255):0.8,
             (71, 51, 3):0.7,
             (0, 0, 0):0.1,
             }



class Node:
    __slots__ = ['x', 'y', 'z', 'g', 'h', 'f', 'parent']

    def __init__(self, x, y, elevation, parent=()):
        self.x = x
        self.y = y
        self.z = elevation
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0
        # self.speed = speed

    def __lt__(self, other):
        return self.elevation < other.elevation

    # Compare nodes
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y



def BFS():
    x = 0
    y = 0
    queue = Queue()
    visited =[]
    # start_node = Node(x, y, elevation_data[x][y], None)
    start_node = x,y
    queue.put(start_node)
    while queue.qsize() != 0:
        current_node = queue.get()
        if current_node not in visited:
            visited.append(current_node)
            print(current_node)
            #print(current_node.x, current_node.y)
            # terraincolor = terrain1.getpixel((current_node.x,current_node.y))
            terraincolor = terrain1.getpixel(current_node)
            #print(terraincolor)
            if terraincolor == (0, 0, 225):
                for i in range(1, 8):
                    if(x+i <= 395 and x-i > 0 and y-i >0 and y+i <= 500):
                        # node1 = Node(x+i,y,elevation_data[x+i][y],current_node)
                        # node2 = Node(x-i,y,elevation_data[x-i][y],current_node)
                        # node3 = Node(x,y+i,elevation_data[x][y-i],current_node)
                        # node4 = Node(x,y-i,elevation_data[x][y-i],current_node)

                        node1 = x+i,y
                        node2 = x-i,y
                        node3 = x,y+i
                        node4 = x,y-i

                        # terraincolor1 = terrain1.getpixel((node1.x, node1.y))
                        # terraincolor2 = terrain1.getpixel((node2.x, node2.y))
                        # terraincolor3 = terrain1.getpixel((node3.x, node3.y))
                        # terraincolor4 = terrain1.getpixel((node4.x, node4.y))
                        terraincolor1 = terrain1.getpixel(node1)
                        terraincolor2 = terrain1.getpixel(node2)
                        terraincolor3 = terrain1.getpixel(node3)
                        terraincolor4 = terrain1.getpixel(node4)

                        # if terraincolor1 == (0, 0, 225) or terraincolor2 == (0, 0, 225) or terraincolor3 == (0, 0, 225) or terraincolor4 == (0 ,0, 225):
                        #     outputImage.putpixel((current_node.x, current_node.y), (191, 239, 255))

                        if terraincolor1 != (0, 0, 255):
                            # queue.put(node2)
                            # queue.put(node3)
                            # queue.put(node4)
                            # outputImage.putpixel((current_node.x, current_node.y), (191, 239, 255))
                            visited.append(node1)
                            outputImage.putpixel(current_node, (191, 239, 255))
                            break
                        elif terraincolor2 != (0, 0, 255):
                            # queue.put(node1)
                            # queue.put(node3)
                            # queue.put(node4)
                            #outputImage.putpixel((current_node.x, current_node.y), (191, 239, 255))
                            outputImage.putpixel(current_node, (191, 239, 255))
                            visited.append(node2)
                            break
                        elif terraincolor3 != (0, 0, 255):
                            # queue.put(node1)
                            # queue.put(node2)
                            # queue.put(node4)
                            # outputImage.putpixel((current_node.x, current_node.y), (191, 239, 255))
                            outputImage.putpixel(current_node, (191, 239, 255))
                            visited.append(node3)
                            break
                        elif terraincolor4 != (0, 0, 255):
                            # queue.put(node1)
                            # queue.put(node2)
                            # queue.put(node3)
                            # outputImage.putpixel((current_node.x, current_node.y), (191, 239, 255))
                            outputImage.putpixel(current_node, (191, 239, 255))
                            visited.append(node4)
                            break
            else:
                #x, y = current_node.x, current_node.y
                x,y =current_node
                neighbors =[(x+1, y), (x-1, y), (x, y-1), (x, y+1)]
                for neighbor in neighbors:
                    x1, y1 = neighbor
                    if x1 >= 0 and y1 >= 0 and x1 < 395 and y1 < 500:
                        new_node = x1,y1
                        #new_node = Node(x1,y1,elevation_data[x1][y1],current_node)
                        if new_node not in visited:
                            queue.put(new_node)
                            #visited.append(new_node)









# def BFS():
#         x = 0
#         y = 0
#         val = False
#         visited=[]
#         queue = Queue()
#         node = Node(x, y, elevation_data[x][y],None)
#         queue.put(node)
#         while queue.qsize() != 0:
#             entry = queue.get()
#             print(entry.x, entry.y)
#             if entry not in visited:
#                 visited.append(entry)
#                 if terrain1.getpixel((entry.x, entry.y)) == (0, 0, 225):
#                     # val = checksevenpixels(entry.x, entry, y)
#                     print(entry.x, entry.y)
#                     return
#                 x, y = entry.x, entry.y
#                 neighbors =[(x+1, y), (x-1, y), (x, y-1), (x, y+1)]
#                 for neighbor in neighbors:
#                     x1, y1 = neighbor
#                     if x1 >= 0 and y1 >= 0:
#                         queue.put(Node(x1,y1,elevation_data[x1][y1],entry))
#
#
#
#
# def checksevenpixels(x,y):
#     neighbors = [(x + 1, y), (x - 1, y), (x, y - 1), (x, y + 1)]
#     flag = 0
#     path=[]
#     for i in range(7):
#         if terrain1.getpixel((x+i,y))!=(0,0,225):
#             flag = 1
#             return False
#         else:
#             path.append((x+i,y))
#
#
#     if(flag!=1):
#         for point in path:
#             x = int(point[0])
#             y = int(point[1])
#             outputImage.putpixel((x, y), (191, 239, 255))
#         outputImage.show()
#     return True







def astar(start_node,end_node):
    closed = []
    visited =[]
    # print("start:",start_node.x,start_node.y)
    # print("end:",end_node.x,end_node.y)
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

        #create neighbors list
        neighbors = [(x+1, y), (x-1, y+1), (x-1, y), (x-1, y-1), (x, y-1), (x+1, y-1), (x, y+1), (x+1, y+1)]
        for neighbor in neighbors:
            #print(neighbor)
            x, y = neighbor
            if x < 395 and y < 500:
                terraincolor = terrain1.getpixel((int(x), int(y)))

                #out of bound terrain check
                if terraincolor == (205,0,101):
                    continue
                mult_factor =checkterraintype(terraincolor)
            else:
                break

            new_entry = Node(neighbor[0], neighbor[1], elevation_data[neighbor[0]] [neighbor[1]], current_node)
            new_entry.g = math.sqrt(pow(new_entry.x-start_node.x, 2)+pow(new_entry.y-start_node.y, 2)+pow(new_entry.z-start_node.z, 2))
            new_entry.h = math.sqrt(pow(new_entry.x-end_node.x, 2)+pow(new_entry.y-end_node.y, 2)+pow(new_entry.z-end_node.z, 2))* mult_factor
            new_entry.f = new_entry.g + new_entry.h
            sortby = new_entry.f

            if add_in_priority_queue(visited, new_entry) == True:
                visited.append(new_entry)
                q.put((sortby, new_entry))

    return None


def add_in_priority_queue(visited,new_entry):
    for node in visited:
        if new_entry == node and new_entry.f >= node.f:
            return False
    return True


def checkterraintype(terraincolor):
    for key in terraintype:
        if key == terraincolor:
            return terraintype[key]
    return -1


def createOutputImage():
    for point in path:
        x = int(point[0])
        y = int(point[1])
        outputImage.putpixel((x, y), (255, 0, 0))


if __name__ == '__main__':

    # index=0
    # while(index < len(events)-1):
    #     start_node = Node(events[index][0], events[index][1], elevation_data[events[index][0]][events[index][1]], None)
    #     end_node = Node(events[index + 1][0], events[index + 1][1],
    #                     elevation_data[events[index + 1][0]][events[index + 1][1]], None)
    #     path.extend(astar(start_node,end_node))
    #     index += 1
    # print(len(path))
    # createOutputImage()
    BFS()
    outputImage.show()
############################################################################################
# def BFS():
#     # x = 149
#     # y = 196
#     x = 233
#     y = 482
#     queue = Queue()
#     visited =[]
#     start_node = x,y
#     queue.put(start_node)
#     while queue.qsize() != 0:
#         current_node = queue.get()
#         if current_node not in visited:
#             visited.append(current_node)
#             print(current_node)
#             terraincolor = terrain1.getpixel(current_node)
#
#             if(terraincolor[0]==0 and terraincolor[1]==0 and terraincolor[2]==255):
#             # if terraincolor == (0, 0, 225):
#                 for i in range(1, 8):
#                     if(current_node[0]+i < 395 and current_node[0]-i > 0 and current_node[1]-i >0 and current_node[1]+i < 500):
#                         node1 = current_node[0]+i, current_node[1]
#                         node2 = current_node[0]-i, current_node[1]
#                         node3 = current_node[0], current_node[1]+i
#                         node4 = current_node[0], current_node[1]-i
#                         terraincolor1 = terrain1.getpixel(node1)
#                         terraincolor2 = terrain1.getpixel(node2)
#                         terraincolor3 = terrain1.getpixel(node3)
#                         terraincolor4 = terrain1.getpixel(node4)
#
#                         if terraincolor1 != (0, 0, 255):
#                             visited.append(node1)
#                             outputImage.putpixel(current_node, (100, 235, 255))
#                             break
#                         else:
#                             queue.put(node1)
#                         if terraincolor2 != (0, 0, 255):
#                             outputImage.putpixel(current_node, (100, 235, 255))
#                             visited.append(node2)
#                             break
#                         else:
#                             queue.put(node2)
#                         if terraincolor3 != (0, 0, 255):
#                             outputImage.putpixel(current_node, (100, 235, 255))
#                             visited.append(node3)
#                             break
#                         else:
#                             queue.put(node3)
#                         if terraincolor4 != (0, 0, 255):
#                             outputImage.putpixel(current_node, (100, 235, 255))
#                             visited.append(node4)
#                             break
#                         else:
#                             queue.put(node4)
#             else:
#                 x, y = current_node
#                 neighbors =[(x+1, y), (x-1, y), (x, y-1), (x, y+1)]
#                 for neighbor in neighbors:
#                     if neighbor not in visited:
#                         x1, y1 = neighbor
#                         if x1 >= 0 and y1 >= 0 and x1 < 395 and y1 < 500:
#                             new_node = x1,y1
#                             print("nn",new_node)
#                             queue.put(new_node)

################################################################################################
# def BFS():
#
#     # visited = []
#     queue = Queue()
#     for i in range(len(edgelist)):
#         queue.put(edgelist[i])
#     while queue.qsize()!=0:
#         current_node = queue.get()
#         x = current_node[0]
#         y = current_node[1]
#         for i in range(1,8):
#             neighbors =[(x+i,y),(x-i,y),(x,y+i),(x,y-i)]
#             for neighbor in neighbors:
#                 x1 = neighbor[0]
#                 y1 = neighbor[1]
#                 if x1>=0 and x1<395 and y1>=0 and y1<500:
#                     neighbor_color = terrain1.getpixel(neighbor)
#                     if(neighbor_color[0]==0 and neighbor_color[1]==0 and neighbor_color[2]==255):
#                         outputImage.putpixel(neighbor,(100,235,255))


#####################################################################################################
#
# def BFS():
#
#     x = 149
#     y = 196
#     start_node = x,y
#     visited =[]
#     queue =Queue()
#     queue.put(start_node)
#     while queue.qsize()!=0:
#         current_node = queue.get()
#         print(current_node)
#         if current_node  not in visited:
#             visited.append(current_node)
#             terraincolor = outputImage.getpixel(current_node)
#             if(terraincolor[0]==0 and terraincolor[1]==0 and terraincolor[2]==255):
#                 x1= current_node[0]
#                 y1=current_node[1]
#                 for i in range(1,8):
#                     neighbors =[(x1+i,y1),(x1-i,y1),(x1,y1-i),(x1,y1-i)]
#                     for neighbor in neighbors:
#                         x1 = neighbor[0]
#                         y1 = neighbor[1]
#                         if x1 >= 0 and x1 < 395 and y1 >= 0 and y1 < 500:
#                             neighbor_color = terrain1.getpixel(neighbor)
#                             if (neighbor_color[0] == 0 and neighbor_color[1] == 0 and neighbor_color[2] == 255):
#                                 outputImage.putpixel(neighbor, (100, 235, 255))
#                             else:
#                                 queue.put(neighbor)
#             else:
#                 x, y = current_node
#                 neighbors =[(x+1, y), (x-1, y), (x, y-1), (x, y+1)]
#                 for neighbor in neighbors:
#                     if neighbor not in visited:
#                         x1, y1 = neighbor
#                         if x1 >= 0 and y1 >= 0 and x1 < 395 and y1 < 500:
#                             new_node = x1,y1
#                             print("nn",new_node)
#                             queue.put(new_node)

###########################################################
# terraintype_summer_winter_spring={(248, 148, 18):0.2,
#              (255, 192, 0):0.5,
#              (255, 255, 255):0.4,
#              (2, 208, 60):0.3,
#              (2, 136, 40):0.6,
#              (5, 73, 24):-1,
#              (0, 0, 255):0.8,
#              (71, 51, 3):0.7,
#              (0, 0, 0):0.1,
#             (100,235,255):0.2,
#             (123,53,27):0.9
#              }



# #create lookup table for terrain in fall
# terraintype_fall={(248, 148, 18):0.2,
#              (255, 192, 0):0.8,
#              (255, 255, 255):0.5,
#              (2, 208, 60):0.4,
#              (2, 136, 40):0.7,
#              (5, 73, 24):-1,
#              (0, 0, 255):0.8,
#              (71, 51, 3):0.9,
#              (0, 0, 0):0.1,
#              }





