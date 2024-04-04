import numpy as np
from sklearn.mixture import GaussianMixture
import random
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import to_categorical
from collections import defaultdict
import matplotlib.pyplot as plt

# Define the size of the pixel group
pixel_size = 100

# means for the normal distributions 
means = np.array([5, 15, 25, 35, 45, 55, 65, 75, 85, 95])

# Shuffle the means for shift
random.shuffle(means)
component_means = np.array([means, means]).T  # For x and y axes

# Variances varies  
variances = np.linspace(1, 5, 10)

# Create multivariate normal distributions for x and y axes with 10 components
gmm_joint = GaussianMixture(n_components=10, covariance_type='diag', random_state=42)

# Generate joint samples for x and y axes
joint_samples = np.stack([np.random.normal(m, v, size=(pixel_size, pixel_size)) for m, v in zip(means, variances)], axis=-1)

# Reshape the joint samples to match the format expected by GMM
joint_samples_reshaped = joint_samples.reshape(-1, 10)

# Fit GMM for the joint distribution of x and y axes
gmm_joint.fit(joint_samples_reshaped)

# Function to calculate the probs for each component at a random point
def calculate_component_probabilities():
    # Select a random point
    x_coord = np.random.randint(pixel_size)
    y_coord = np.random.randint(pixel_size)
    
    # Calc joint probs using the joint GMM
    point = joint_samples[x_coord, y_coord].reshape(1, -1)
    joint_probs = gmm_joint.predict_proba(point)[0]
    
    # Print probs for each component
    print(f"Probabilities for point ({x_coord}, {y_coord}):")
    for component, prob in enumerate(joint_probs):
        print(f"Component {component}: Joint probability: {prob:.4f}")

# random point probability printing
calculate_component_probabilities()  #########################################check point

#client allocate into pixels 

client_num=50
# put randomly selected x_coord,y_coord list sets
clients=[] 
for i in range(client_num):
    x_coord=np.random.randint(pixel_size)
    y_coord=np.random.randint(pixel_size)
    clients.append((x_coord,y_coord))
    
#function to select digits of each pixels using probs as weight
def select_digit(x,y):
    
    point = joint_samples[x, y].reshape(1, -1)
    joint_probs = gmm_joint.predict_proba(point)[0]
    
    selected_digit= random.choices(range(10),weights=joint_probs, k=1)[0]
    return selected_digit

client_digits=[]
print("Clients coordinates shown")
print(clients) # check point for clients ##########################################

#allocate each clients of one label
for i in range(client_num):
    point=clients[i]
    client_digits.append(select_digit(point[0],point[1]))

print("Clients digit to have ")
print(client_digits) #check point for clients digit ####################################

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to the range [0, 1] (normalization)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float64') / 255.0

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test,num_classes=10)

#dataset = (x,y) type assume!!!!!!!!!!!!!!!!!!!!!
def select_mnist_datasets(label,dataset,num_samples):
    selected_datasets=defaultdict(list)
    indices=np.where(np.argmax(dataset[1],axis=1)==label)[0]
    sampled_indices=np.random.choice(indices,size=num_samples,replace=False)
    selected_datasets=dataset[0][sampled_indices]
    
    return selected_datasets

#allocate 100 datasets for each clients
clients_datasets=[]

for i in range(50):
    client_label=client_digits[i]
    clients_datasets.append(select_mnist_datasets(client_label,dataset=(x_train,y_train),num_samples=100))

#check point ########################################### client datasets allocation ####################################
print("Check point for clients_datasets set well for 100 datas each")
print(len(clients_datasets[3]))
print(len(clients_datasets[30]))

#1. Edge IID setting 
"""
each clients 1 class only
edge server have 10 clients with different classes 

"""
### I make each client to have 100 datasets bcz in iid setting it will randomly make 50 clients all to have same label

# Dividing clients into edge servers (each server with 10 clients)
edge_servers = [] 
clients_edge_point=[]

for i in range(10):
    selected_edge=np.random.choice(50,size=5,replace=False) #client_digits
    for i in selected_edge:
        edge_servers.append(clients_datasets[i])
        clients_edge_point.append(clients[i])
        

####check point for edge_server dividing ##########################################
print("Check point for edge_server well divided")
print(len(edge_servers[0]))
print(len(edge_servers[4]))

print(len(edge_servers))


# Ensure the lengths are within bounds before plotting
print("Lengths for edge servers and edge points:", len(edge_servers), len(clients_edge_point))

# Plotting the clients and differentiating them based on edge servers with different colors
plt.figure(figsize=(8, 6))

colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta', 'brown', 'lime']  # Define colors for each edge server
for i, edge_server in enumerate(clients_edge_point):
    x_coord= edge_server[0]
    y_coord=edge_server[1]
    plt.scatter(x_coord, y_coord, color=colors[i // 10]) # label=f'Edge Server {i // 10}'
    #plt.text(x_coord, y_coord, client_digits[i], fontsize=12, ha='right', va='bottom')  # Add the label value beside each point

plt.title('Clients Assigned to Edge Servers Edge IID Setting')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.show()




