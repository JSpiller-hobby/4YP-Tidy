#SYNTHETIC DATA GENERATION - GEARS.
#A script to generate training data. Data comprises JPG images of 2D objects made of circles. Every pixel has a velocity, stored as a h*w*2 numpy array for each image. 
#The velocities are calculated as if the circles were meshing spur gears (simplified), and meshing is represented by circels just touching at the edge; no intersection.
#Images include random number of gears; velocity magnitude; arrangement of gears; size of gears. One assembly per image - no detached gears. Loops of gears are not
#possible in this data.
#
#Jack Spiller
#Following outline from Andrea Vedaldi with help from Yushi LAN.
#Uses code from https://github.com/phyworld/phyworld - subject to Apache 2.0 license, so this file is also licensed under the Apache 2.0 license.
#Copies of this license can be found here https://www.apache.org/licenses/LICENSE-2.0
#Created: 31/10/25
#Last modified: 06/11/25
#
#TODO: fix the bug causing generation to get stuck. I think a loop counter which restarts the function at the current gear index if there are too many attempts will work.

#IMPORTS
import numpy as np
import random
from PIL import Image
import json
import os

#PARAMETERS
path = ".//data//1GearData//"
n_data = 10000
width = 256
min_rad = width/10 
max_rad = width/3
min_gears = 1
max_gears = 1
edge_width = 0.03
velocity_scale = 1

#REUSABLE CODE
def find_velocity(r, x, y, w, width):
    rot_dir = np.sign(w)
    rot_mag = abs(w)

    centre = np.array([x, y])
    X, Y = np.meshgrid(range(width), range(width))
    dx = X - centre[0, np.newaxis, np.newaxis]                     
    dy = Y - centre[1, np.newaxis, np.newaxis]  
    distance_squared = dx**2 + dy**2  

    v_x = rot_dir*rot_mag*(centre[1, np.newaxis, np.newaxis] - Y)
    v_y = rot_dir*rot_mag*(X - centre[0, np.newaxis, np.newaxis])
    for j in range(width):
        for i in range(width):
            if distance_squared[i,j] > r**2:
                v_x[i,j] = 0
                v_y[i,j] = 0

    velocity_field = np.append(v_x[:,:,None], v_y[:,:,None], axis = 2) 

    return velocity_field



def next_gear(V, W):
    feasible = False

    while feasible == False:
        check = 0 

        # 4) Random index from uniform(0, ... ,V-1), where V is the list of gears, to choose the gear to which the next is added.
        idx = int(random.uniform(0, len(V))) #NOTE: may error if exactly len(V) is chosen, but highly improbable.

        r, x, y = V[idx, 0:3]
        w = W[idx]

        # 5) Next radius from uniform distr
        r_prime = random.uniform(min_rad, max_rad)

        # 6) Random theta between 0 and 2pi
        theta = random.uniform(0, 2*np.pi)

        # 7) Next x coord = x +(r1+r2)cos(theta)
        x_prime = x + (r + r_prime)*np.cos(theta)

        # 8) Next y coord = y +(r1+r2)sin(theta)
        y_prime = y + (r + r_prime)*np.sin(theta)

        # 9) Feasibility test: for all (x,y,r) in V: does the distance between centres exceed the sum of radii (+spacing) and stay wihtin image bounds?
        if x_prime - r_prime > 0 and y_prime - r_prime >0 and x_prime + r_prime < width and y_prime + r_prime < width:
            for (r_f, x_f, y_f) in V:
                if x_f == x and y_f == y:
                    pass
                    check += 1
                else:
                    
                    if (x_prime - x_f)**2 + (y_prime - y_f)**2 > (r_f + r_prime + 20)**2:
                        check += 1
            
        if check == len(V):
            feasible = True
        
    # 10) Add new (r,x,y) to V
    next_gear = np.array([[r_prime, x_prime, y_prime]])
    V = np.append(V, next_gear, axis = 0)

    #calculate pixel-wise velocities
    w = -1*r*w/r_prime
    velocity_field = find_velocity(r_prime, x_prime, y_prime, w, width)

    W = np.append(W, w)

    return V, W, idx, velocity_field





def make_gear_image(r, x, y, width, edge_width):
    colour = (random.random()*255, random.random()*255, random.random()*255)

    rad2sq = r**2
    centre = np.array([x, y])
    make_touching = 10

    X, Y = np.meshgrid(range(width), range(width))
    dx = X - centre[0, np.newaxis, np.newaxis]                     
    dy = Y - centre[1, np.newaxis, np.newaxis]  
    distance_squared = dx**2 + dy**2  
    gear_mask = np.clip(((rad2sq + make_touching) - distance_squared) / (rad2sq * edge_width), 0, 1)[:,:,None]

    image = np.ones((width, width, 3), dtype=np.uint8) * 0
    gear_image = (1 - gear_mask) * image + gear_mask * colour 

    return gear_image

def generate(datum, width, min_rad, max_rad, min_gears, max_gears, edge_width, velocity_scale, path):
    #Radius from uniform distribution
    r1 = random.uniform(min_rad, max_rad)

    #Centre x from uniform distribution
    x1 = random.uniform(0 + r1, width - r1)

    #Centre y from uniform distribution -> store (centre, radius) in V 
    y1 = random.uniform(0 + r1, width - r1)

    #Angular velocity from unifiorm distribution
    w = random.random() * velocity_scale

    #create the graph 
    n_gears = int(random.uniform(min_gears, max_gears + 1))

    V = np.array([[r1, x1, y1]])
    E = np.zeros((n_gears, n_gears))
    W = np.array([w])

    velocity1 = find_velocity(r1, x1, y1, w, width)
    velocities = [velocity1]

    #generate subsequent gears

    g = 1
    while g < n_gears:
        V, W, idx, velocity_field = next_gear(V, W)
        velocities.append(velocity_field)
        #Populate E, the edges, an adjacency matrix. G = (V,E)
        E[g, idx] = 1
        g += 1

    E = E + E.T

    #assign the image and velocity (mask) folders

    image_folder = os.path.join(path, "imgs")
    mask_folder = os.path.join(path, "masks")

    #Make a folder for graphs 

    graph_folder = os.path.join(path, "graphs")

    #Put the graph (V,E pairs) in a JSON file
    structure = {"adjacency": {}, "gears": {}}

    for row in range(len(E)):
        data = np.ndarray.tolist(E[row,:])
        name = f"gear{row}"
        structure["adjacency"].update({name: data})

    for gear in range(len(V)):
        list = np.ndarray.tolist(V[gear,:]) 
        data = {}
        data.update({"radius": list[0]})
        data.update({"centre x-coord": list[1]})
        data.update({"centre y-coord": list[2]})

        name = f"gear{gear}"
        structure["gears"].update({name: data})

    json_file = os.path.join(graph_folder, f"obj{datum}.json")
    with open(json_file, "w") as outfile:
        json.dump(structure, outfile, indent=4)

    #save the image as a JPG.
    gear_images = []
    for (r, x, y) in V:
        gear_image = make_gear_image(r, x, y, width = width, edge_width = edge_width)
        gear_images.append(gear_image)

    sum_image = sum(gear_images)
    img = Image.fromarray(sum_image.astype(np.uint8))

    image_file = os.path.join(image_folder, f"obj{datum}.jpg")
    img = img.save(image_file)

    #Save the velocities as a .npy array
    sum_velocity = sum(velocities)

    vel_file = os.path.join(mask_folder, f"obj{datum}.npy")
    with open(vel_file, 'wb') as f:
        np.save(f, sum_velocity)


#BODY
#NOTE: for now, if it gets stuck, just change the starting index below to the index after the most recent successful datum, then restart this script.
for datum in range(0, n_data, 1):
    generate(datum, width=width, min_rad=min_rad, max_rad=max_rad, min_gears=min_gears, max_gears=max_gears, edge_width=edge_width, velocity_scale=velocity_scale, path=path)