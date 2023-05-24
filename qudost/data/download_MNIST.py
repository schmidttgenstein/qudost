import os
import gzip
import shutil
import urllib.request
import numpy as np
from PIL import Image
def download_mnist_data():
    # Define the URLs for the MNIST dataset
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    file_names = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                  't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    download_dir = './mnist_data/'

    # Create the download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)

    # Download the MNIST dataset files
    for file_name in file_names:
        file_url = base_url + file_name
        file_path = os.path.join(download_dir, file_name)
        urllib.request.urlretrieve(file_url, file_path)

        # Extract the downloaded files
        with gzip.open(file_path, 'rb') as f_in:
            extract_path = file_path[:-3]  # Remove the '.gz' extension
            with open(extract_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove the compressed file
        os.remove(file_path)

    # Load the dataset into NumPy arrays
    train_images = _load_images(download_dir, 'train')
    train_labels = _load_labels(download_dir, 'train')
    test_images = _load_images(download_dir, 't10k')
    test_labels = _load_labels(download_dir, 't10k')

    # Organize the images into folders by label
    _organize_images_by_label(train_images, train_labels, download_dir, 'train')
    _organize_images_by_label(test_images, test_labels, download_dir, 'test')

    print("MNIST dataset downloaded and organized successfully.")

def _load_images(download_dir, name):
    file_path = os.path.join(download_dir, f'{name}-images-idx3-ubyte')
    with open(file_path, 'rb') as f:
        # Read the header information
        magic_number = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')

        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows, num_cols)
    
    return images

def _load_labels(download_dir, name):
    file_path = os.path.join(download_dir, f'{name}-labels-idx1-ubyte')
    with open(file_path, 'rb') as f:
        # Read the header information
        magic_number = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')

        # Read the label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    return labels

def _organize_images_by_label(images, labels, download_dir, name):
    output_dir = os.path.join(download_dir, f'{name}_data')
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories for each label
    for label in range(10):
        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)

    # Save images into label-specific folders
    for i, (image, label) in enumerate(zip(images, labels)):
        label_dir = os.path.join(output_dir, str(label))
        image_path = os.path.join(label_dir, f'{i}.png')
        image = (255 - image).astype(np.uint8)  # Invert colors (black background)
        image = Image.fromarray(image)
        image.save(image_path)

    print(f'{name.capitalize()} data organized by label.')

# Call the function to download and organize the MNIST dataset
download_mnist_data()