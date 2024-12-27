import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import configs
import numpy as np

motion_dir = configs.MOTION_PATH
good_dir = configs.GOOD_PATH

# motion_test_dir = configs.MOTION_TEST_PATH
# good_test_dir = configs.GOOD_TEST_PATH

size = configs.IMG_SIZE
batch_size = configs.BATCH_SIZE

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size),
    transforms.Lambda(lambda t: (t * 2) - 1),
    transforms.RandomHorizontalFlip(),
    transforms.Grayscale(num_output_channels=1),
])
reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
        transforms.ToPILImage(),
])

motion_dataset = ImageFolder(motion_dir, transform=transform)
good_dataset = ImageFolder(good_dir, transform=transform)
# motion_test_dataset = ImageFolder(motion_test_dir, transform=transform)
# good_test_dataset = ImageFolder(good_test_dir, transform=transform)

motion_loader = DataLoader(motion_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
good_loader = DataLoader(good_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# motion_test_loader = DataLoader(motion_test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# good_test_loader = DataLoader(good_test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
