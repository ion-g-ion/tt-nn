import torch as tn
from torchvision import datasets, transforms
import torchtt as tntt

# import dataset
data_dir = 'chest_xray/'
transform = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224), transforms.ToTensor()])
dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader_train = tn.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)