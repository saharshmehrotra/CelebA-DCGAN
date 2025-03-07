import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image, 0  # Return 0 as a dummy label

def get_celeba_loader(data_path='./data', batch_size=128, num_workers=2):
    """
    Create a DataLoader for the CelebA dataset with appropriate transformations.
    """
    # Define the image transformations
    transform = transforms.Compose([
        transforms.CenterCrop(178),  # CelebA specific center crop
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
    ])

    # Create dataset
    dataset = CelebADataset(
        root_dir=os.path.join(data_path, 'celeba', 'images', 'img_align_celeba'),
        transform=transform
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader