import numpy as np
import matplotlib.pyplot as plt
import random

matrix_size = 100

# Mean values digit 0~9 임의 설정 
digit_means_x = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
digit_means_y = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]

random.shuffle(digit_means_x)
random.shuffle(digit_means_y)

#  multivariate normal distributions 
x = np.zeros((matrix_size, matrix_size))
y = np.zeros((matrix_size, matrix_size))

for i in range(matrix_size):
    for j in range(matrix_size):
        # random digit label 선택 
        digit_x = np.random.randint(0, 10)
        digit_y = np.random.randint(0, 10)
        
        #  multivariate normal distribution x,y each
        mean_x = digit_means_x[digit_x]
        mean_y = digit_means_y[digit_y]
        cov_matrix = np.eye(2) * 5  # Covariance matrix 
        
        sample = np.random.multivariate_normal([mean_x, mean_y], cov_matrix)
        x[i, j] = sample[0]
        y[i, j] = sample[1]

# scatter plot 
plt.figure(figsize=(8, 6))
plt.scatter(x, y, s=20, alpha=0.5)
plt.title('Multivariate Normal Distribution for X and Y Axes')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.show()

