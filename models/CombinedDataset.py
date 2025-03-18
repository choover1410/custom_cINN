import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset


class CombinedDataset(Dataset):
    def __init__(self, image_dir, timeseries_dir, transform=None):
        """
        Args:
            image_dir (str): Directory containing image files.
            timeseries_dir (str): Directory containing time series text files.
            transform (callable, optional): Optional transform to be applied on the images.
        """
        self.image_dir = image_dir
        self.timeseries_dir = timeseries_dir
        self.transform = transform

        # Get sorted list of image and timeseries file paths
        self.image_paths = sorted([
            os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        self.timeseries_paths = sorted([
            os.path.join(timeseries_dir, fname) for fname in os.listdir(timeseries_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        assert len(self.image_paths) == len(self.timeseries_paths), (
            f"Number of images ({len(self.image_paths)}) and time series files ({len(self.timeseries_paths)}) must match"
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and transform the image
        x = Image.open(self.image_paths[idx]).convert("1")
        if self.transform:
            x = self.transform(x)
        
        y = Image.open(self.timeseries_paths[idx]).convert("1")
        if self.transform:
            y = self.transform(y)

        return x, y
