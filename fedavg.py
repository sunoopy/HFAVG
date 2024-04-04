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
import csv 
import matplotlib.pyplot as plt

#import fedavg

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
print(clients)


#allocate each clients of one label
for i in range(client_num):
    point=clients[i]
    client_digits.append(select_digit(point[0],point[1]))

print("Clients digit to have ")
print(client_digits)

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
clients_label=[]


for i in range(50):
    client_label=client_digits[i]
    clients_label.append(client_label)
    clients_datasets.append(select_mnist_datasets(client_label,dataset=(x_train,y_train),num_samples=100))


# Dividing the 100x100 grid into 4 groups of 50x50 size
group_indices = [
    ((0, 50), (0, 50)),     # Top-left quadrant
    ((0, 50), (50, 100)),   # Top-right quadrant
    ((50, 100), (0, 50)),   # Bottom-left quadrant
    ((50, 100), (50, 100))  # Bottom-right quadrant
]

# Calculating the centers of each group
centers = [
    (25, 25),    # Top-left quadrant center
    (25, 75),    # Top-right quadrant center
    (75, 25),    # Bottom-left quadrant center
    (75, 75)     # Bottom-right quadrant center
]

colors = ['red', 'blue', 'green', 'orange']
plt.figure(figsize=(8, 6))

# Scatter plot of client coordinates
for i, coord in enumerate(clients):
    x_coord, y_coord = coord
    color = None

    for group_idx in group_indices:
        if group_idx[0][0] <= x_coord < group_idx[0][1] and group_idx[1][0] <= y_coord < group_idx[1][1]:
            color = colors[group_indices.index(group_idx)]
            break

    plt.scatter(x_coord, y_coord, marker='o', color=color)

# Mark group centers
for center, color in zip(centers, colors):
    plt.scatter(center[0], center[1], marker='x', color=color, s=100)

plt.title('Coordinates of 50 Clients with Different Pixel Colors')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.show()


# Define your model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(100, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Function to train a model on client data
def train_model_on_data(model, data, epochs=5):
    x, y = data
    x = x.reshape((-1, 28, 28, 1))

    model.fit(x, y, epochs=epochs, verbose=0)
    return model

# Federated averaging process
accuracy_threshold = 0.97
rounds = 0
accuracy_progress = []
consecutive_rounds_no_change = 0
previous_accuracy = 0

while True:
    round_accuracy = []

    # Train on each group of clients
    for i, group_idx in enumerate(group_indices):
        group_center = centers[i]
        group_clients = [client for client in clients if group_idx[0][0] <= client[0] < group_idx[0][1] and group_idx[1][0] <= client[1] < group_idx[1][1]]

        # Initialize group datasets
        group_x = []
        group_y = []
        

        # Collect data for each client within the group
        for client in group_clients:
            client_data = clients_datasets[clients.index(client)]
            client_label = clients_label[clients.index(client)]

            # Append client data to group datasets
            group_x.extend(client_data)
            group_y.extend([to_categorical(client_label, num_classes=10)] * len(client_data))

        # Convert group data to numpy arrays
        group_x = np.array(group_x)
        group_y = np.array(group_y)

        # Train model on data from group clients
        model = train_model_on_data(model, (group_x, group_y))

        # Evaluate accuracy on test data after training for each client group
        test_loss, test_accuracy = model.evaluate(x_test.reshape((-1, 28, 28, 1)), y_test, verbose=0)
        round_accuracy.append(test_accuracy)

    rounds += 1
    avg_accuracy = np.mean(round_accuracy)
    accuracy_progress.append(avg_accuracy)

    print(f"Round {rounds} - Average Accuracy: {avg_accuracy}")
    # Check if the accuracy doesn't change
    if np.isclose(avg_accuracy, previous_accuracy):
        consecutive_rounds_no_change += 1
    else:
        consecutive_rounds_no_change = 0

    # Update the previous accuracy
    previous_accuracy = avg_accuracy

    # Check stopping condition
    if consecutive_rounds_no_change >= 5:
        print(f"No improvement in accuracy for {consecutive_rounds_no_change} rounds.")
        break

    # Check stopping condition
    """
    if avg_accuracy >= accuracy_threshold:
        print("Training stopped - desired accuracy reached.")
        break
        """
with open('accuracy_progress4.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Round', 'Accuracy'])
    for i in range(len(accuracy_progress)):
        writer.writerow([i + 1, accuracy_progress[i]])
# Plotting accuracy progress
plt.figure(figsize=(8, 6))
plt.plot(range(1, rounds + 1), accuracy_progress, marker='o')
plt.title('Accuracy Progress')
plt.xlabel('Rounds')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()


