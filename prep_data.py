import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

#I won't normalize for interpretability, but we could try it for performance reasons later
dataprep = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean, std),
])

def prep_data():
    """Downloads MNIST to the data folder if it is not present, and puts it into a DataLoader with batch_size = 128 for easy access. Returns a tuple (train_loader, test_loader)"""
    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=dataprep)
    test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=dataprep)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    return train_loader, test_loader