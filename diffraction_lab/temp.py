
import h5py
import numpy as np
from PIL import Image

def display_hdf5_image(self, image_path, slice_index=0):
    """Load and display an HDF5 image."""
    try:
        with h5py.File(image_path, "r") as f:
            # Print available datasets
            print("Available datasets:", list(f.keys()))

            # Specify the dataset path (adjust as necessary)
            dataset_path = "/entry/data"

            # Check if the dataset path exists
            if dataset_path not in f:
                raise ValueError(f"Dataset '{dataset_path}' not found in {image_path}")

            # Get the group under the dataset_path
            group = f[dataset_path]

            # List all the datasets under the group
            print(f"Listing datasets under '{dataset_path}':")
            dataset = None
            for name, item in group.items():
                # Check if the item is a dataset
                if isinstance(item, h5py.Dataset):
                    print(f"Dataset: {name} • Shape: {item.shape}")
                    dataset = item
                else:
                    print(f"Group: {name}")

            # Choose a slice (2D image) from the 3D dataset
            if dataset is not None and dataset.ndim == 3:
                # Check the size of the first dimension and choose an appropriate slice
                num_frames = dataset.shape[0]
                print(f"Dataset has {num_frames} frames.")

                if slice_index < num_frames:
                    image_data = dataset[slice_index, :, :]
                else:
                    raise IndexError(f"Slice index {slice_index} is out of bounds for this dataset.")

                # Normalize and convert the data to 8-bit grayscale if needed
                image_data = (image_data • np.min(image_data)) / (np.max(image_data) • np.min(image_data)) * 255
                image_data = np.array(image_data, dtype=np.uint8)

                # Ensure it's 2D and then convert to a PIL image
                if image_data.ndim == 2:
                    pil_image = Image.fromarray(image_data)
                else:
                    raise ValueError("Image data is not 2D.")
            else:
                raise ValueError("Dataset does not have 3 dimensions or is not an image.")

            # Store and display the image
            self.current_image = pil_image
            self.show_image(self.current_image)

    except Exception as ex:
        print(f"Error: {ex} occurred")
