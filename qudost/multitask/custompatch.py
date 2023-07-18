import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class CustomFilterGenerator:
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def create_custom_filter(self, shape_type, center_size=None):
        # Create a filter tensor of size (1 x patch_size x patch_size) filled with ones
        filter_size = (1, self.patch_size, self.patch_size)
        filter_data = -torch.ones(filter_size)

        if shape_type == 'horizontal_line':
            
            filter_data[:, :, :] = -1.0
            
            filter_data[:, self.patch_size // 2, :] = 1.0

        elif shape_type == 'vertical_line':
            # Set all pixels to white
            filter_data[:, :, :] = -1.0
            # Set the middle column to black
            filter_data[:, :, self.patch_size // 2] = 1.0

        elif shape_type == 'plus_sign':
            filter_data[:,:,:] = -1.0
            filter_data[:,self.patch_size // 2, :] = 1.0
            filter_data[:,:,self.patch_size // 2] = 1.0

        elif shape_type == 'circle':
            # Create a circular filter with a thin black border, white inside the boundary, and white outside
            center_x = self.patch_size // 2
            center_y = self.patch_size // 2
            radius = self.patch_size // 2 - 2  # Adjust the radius to create a thinner border

            x, y = np.ogrid[:self.patch_size, :self.patch_size]
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2

            if center_size is not None:
                # Adjust the size of the center by modifying the mask
                center_start = center_x - center_size // 2
                center_end = center_start + center_size
                mask[center_start:center_end, center_start:center_end] = True

            filter_data[:, mask] = 1.0

        elif shape_type == 'semi_circle':
            # Create a semicircular filter with a thin black border, white inside the boundary, and white outside
            center_x = self.patch_size // 2
            center_y = self.patch_size // 2
            radius = self.patch_size // 2 - 2  # Adjust the radius to create a thinner border

            x, y = np.ogrid[:self.patch_size, :self.patch_size]
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
            mask &= y >= center_y

            if center_size is not None:
                # Adjust the size of the center by modifying the mask
                center_start = center_x - center_size // 2
                center_end = center_start + center_size
                mask[center_start:center_end, center_start:center_end] = True

            filter_data[:, mask] = 1.0

        elif shape_type == 'right_angle':
            filter_data[:, :, :] = -1.0
            filter_data[:, self.patch_size - 1, :] = 1.0  # Set the last row to 1
            filter_data[:, :, self.patch_size - 1] = 1.0  # Set the last column to 1
        
        elif shape_type == 'plus_sign_no_right':
            filter_data[:, :, :] = -1.0
            filter_data[:, self.patch_size // 2, :] = 1.0
            filter_data[:, :, self.patch_size // 2] = 1.0
            filter_data[:, self.patch_size // 2:, :] = -1.0
        elif shape_type == 'u_shape':
            filter_data[:, :, :] = -1.0
            filter_data[:, :, 0] = 1.0  # Set the first column to 1
            filter_data[:, :, -1] = 1.0  # Set the last column to 1
            filter_data[:, -1, :] = 1.0  # Set the last row to 1
        
        elif shape_type == 'custom':
            filter_data[:, :, :] = -1.0
            filter_data[:, :self.patch_size // 2, 0] = 1.0
            filter_data[:, -1, 0] = 1.0  # Set the last row of the first column to 1
            filter_data[:, :self.patch_size // 2, -1] = 1.0
            filter_data[:, -1, -1] = 1.0  # Set the last row of the last column to 1
            filter_data[:, self.patch_size // 2, :] = -1.0
        else:
            raise ValueError("Invalid shape type. Available options are 'horizontal_line', 'vertical_line', 'circle', 'semi_circle', and 'right_angle'.")

        # Convert the filter to a torch tensor
        filter_tensor = torch.FloatTensor(filter_data)

        # Normalize the filter
        #filter_tensor = (filter_tensor - 0.5) / 0.5

        return filter_tensor

'''
# Example usage
patch_size = 5
shape_type = 'right_angle'
#center_size = 10  # Adjust the size of the white center

filter_generator = CustomFilterGenerator(patch_size)
filter_tensor = filter_generator.create_custom_filter(shape_type)

print(filter_tensor)


# Plot the image of the filter patch
plt.imshow(filter_tensor[0], cmap='gray')
plt.axis('off')
plt.show() '''