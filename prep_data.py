import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# 1. --- Define the new Dataset class ---
# This class inherits from MNIST and modifies __getitem__
# to return (data, label, index)
class MNISTWithIndices(torchvision.datasets.MNIST):
    def __getitem__(self, index):
        # Get the data and target from the parent class (which handles transforms)
        img, target = super().__getitem__(index)

        # Return the image, target, and the index
        return img, target, index

# --- Your original code continues from here ---

#I won't normalize for interpretability, but we could try it for performance reasons later
dataprep = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean, std),
])

def prep_data():
    """Downloads MNIST to the data folder if it is not present, and puts it into a DataLoader with batch_size = 128 for easy access. Returns a tuple (train_loader, test_loader)"""

    # 2. --- Use the new class instead of the default MNIST ---
    train_dataset = MNISTWithIndices(
        './data', train=True, download=True, transform=dataprep
    )
    test_dataset = MNISTWithIndices(
        './data', train=False, download=True, transform=dataprep
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    return train_loader, test_loader