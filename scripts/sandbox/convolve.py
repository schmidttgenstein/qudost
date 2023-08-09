import numpy as np

# Define the matrix and the kernel
four = np.array([
    [0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 1, 1],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0]
])

one = np.array([[0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0]])

kernel = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
])

# Get the dimensions of the matrix and the kernel
matrix_height, matrix_width = four.shape
kernel_height, kernel_width = kernel.shape

# Define the result matrix
result = np.zeros((matrix_height, matrix_width))

# Perform the convolution
for i in range(matrix_height - kernel_height + 1):
    for j in range(matrix_width - kernel_width + 1):
        matrix_subset = four[i:i+kernel_height, j:j+kernel_width]
        result[i+1, j+1] = np.sum(matrix_subset * kernel)

relu_result = np.maximum(result, 0)
mean_value = np.mean(relu_result)
# Print the result
print("convolution = ", result)
print("RuLU = ", relu_result)
print("Mean Output = ", mean_value)
