from AttnUnet import Unet
from PlainUnet import SimpleUnet_plain
from Discriminator import Discriminator
from torch.optim import Adam
import configs 
from loaders import motion_loader, good_loader
from diffusion import train_one_epoch
import torch

device = configs.DEVICE
CHECKPOINT_DIR = configs.CHECKPOINT_DIR

def get_model(plain=False):
    if not plain:
        model = Unet(
            configs.DIM,
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

    #model 1 - generate motion
    motion_gen = get_model().to(device)
    #model 2 - generate good
    good_gen = get_model().to(device)
    #model 3 - motion -> good
    motion2good = get_model(plain=True).to(device)
    #model 4 - good -> motion
    good2motion = get_model(plain=True).to(device)
    netD_A = Discriminator(input_nc=1).to(device)
    netD_B = Discriminator(input_nc=1).to(device)

    #optimizers 
    optim_gen = Adam(list(motion_gen.parameters())+list(good_gen.parameters()), lr=configs.LR, betas=configs.BETA)
    optim_cyc = Adam(list(motion2good.parameters())+list(good2motion.parameters()), lr=configs.LR, betas=configs.BETA)
    optim_d = Adam(list(netD_A.parameters()) + list(netD_B.parameters()), lr=configs.LR, betas=configs.BETA)

    if configs.LOAD_FROM_CHECKPOINT:
        motion_gen.load_state_dict(torch.load(CHECKPOINT_DIR+'motion.pt'))
        good_gen.load_state_dict(torch.load(CHECKPOINT_DIR+'good.pt'))
        motion2good.load_state_dict(torch.load(CHECKPOINT_DIR+'motion2good.pt'))
        good2motion.load_state_dict(torch.load(CHECKPOINT_DIR+'good2motion.pt'))
        netD_A.load_state_dict(torch.load(CHECKPOINT_DIR+'netD_A.pt'))
        netD_B.load_state_dict(torch.load(CHECKPOINT_DIR+'netD_B.pt'))

    models = [
        motion_gen,
        good_gen,
        motion2good,
        good2motion,
        netD_A,
        netD_B
    ]

    optimizers = [
        optim_gen,
        optim_cyc,
        optim_d
    ]

    for ep in range(1, configs.EPOCHS):
        train_one_epoch(ep, models, optimizers, motion_loader, good_loader, identity=True)
        epoch_checkpoint_path = CHECKPOINT_DIR + 'good_epoch{}.pt'.format(ep)
        torch.save(good_gen.state_dict(), epoch_checkpoint_path)

 