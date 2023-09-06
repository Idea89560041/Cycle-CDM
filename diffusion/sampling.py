from diffusion import translate
from AttnUnet import Unet
from PlainUnet import SimpleUnet_plain
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import torchvision.transforms as transforms

RELEASE_TIME = 100
CHECKPOINT_DIR = "./checkpoint/artefact_free.pt"
INPUT_DIR = "./data/artefact_corrupted"
OUTPUT_DIR = "./data/artefact_free"
DEIVCE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(96),
    transforms.Lambda(lambda t: (t * 2) - 1),
])

reverse_transforms = transforms.Compose([
    transforms.Lambda(lambda t: (t + 1) / 2),
    transforms.Lambda(lambda t: t.permute(1, 2, 0)),
    transforms.Lambda(lambda t: t * 255.),
    transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
])

def get_model(plain=False):
    if not plain:
        model = Unet(
            dim=32,
            channels=2,
            out_dim=1,
            dim_mults=(1, 2, 4, 8, 8),
        )
    else:
        model = SimpleUnet_plain(
            in_dim=1,
            dim=64,
        )
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    return model

if __name__=="__main__":
    artefact_corrected = get_model().to(DEIVCE)
    artefact_corrected.load_state_dict(torch.load(CHECKPOINT_DIR))
    artefact_corrected.eval()

    image_files = os.listdir(INPUT_DIR)
    images = []
    for file in image_files:
        image_path = os.path.join(INPUT_DIR, file)
        image = plt.imread(image_path)
        image = transform(image)
        images.append((file, image))

    tensor_images = [img[1] for img in images]
    tensor_images = torch.stack(tensor_images).to(DEIVCE)

    with torch.no_grad():
        pred_artefact_free = translate(tensor_images, artefact_corrected, RELEASE_TIME)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i, (filename, image) in enumerate(zip(image_files, pred_artefact_free)):
        output_filename = filename[:-4] + ".png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        image = reverse_transforms(image)
        image.save(output_path)

