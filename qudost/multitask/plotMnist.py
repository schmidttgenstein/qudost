import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms

# Set the number of images to plot and the digit to display
N = 5  # Number of images
K = 4  # Digit to display


transf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST dataset
mnist_train_dataset = MNIST(root='./data', train=True, transform=transf, download=True)

# Filter the dataset to obtain images of digit K
filtered_indices = [i for i in range(len(mnist_train_dataset)) if mnist_train_dataset.targets[i] == K]
filtered_images = [mnist_train_dataset[i][0] for i in filtered_indices[:N]]

# Print the tensors and plot the images
fig, axes = plt.subplots(1, N, figsize=(10, 4))
for i, ax in enumerate(axes):
    image = filtered_images[i]
    print(f"Tensor shape: {image.shape}")
    print(f"Tensor values:\n{image}")
    image = (image * 0.5) + 0.5  # Denormalize the image
    ax.imshow(image.squeeze(), cmap='gray')
    ax.axis('off')

plt.show()
