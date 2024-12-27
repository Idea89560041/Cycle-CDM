from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import configs
import torchvision.models as models
from torch.autograd import Variable
import torch
from torchvision.models import VGG16_Weights
from Discriminator import ReplayBuffer

device = configs.DEVICE
e = configs.RELEASE_TIME

class VGG16(nn.Module):
  def __init__(self):
    super(VGG16, self).__init__()
    features = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
    self.relu1_1 = torch.nn.Sequential()
    self.relu1_2 = torch.nn.Sequential()

    self.relu2_1 = torch.nn.Sequential()
    self.relu2_2 = torch.nn.Sequential()

    self.relu3_1 = torch.nn.Sequential()
    self.relu3_2 = torch.nn.Sequential()
    self.relu3_3 = torch.nn.Sequential()

    self.relu4_1 = torch.nn.Sequential()
    self.relu4_2 = torch.nn.Sequential()
    self.relu4_3 = torch.nn.Sequential()

    self.relu5_1 = torch.nn.Sequential()
    self.relu5_2 = torch.nn.Sequential()
    self.relu5_3 = torch.nn.Sequential()

    for x in range(2):
      self.relu1_1.add_module(str(x), features[x])

    for x in range(2, 4):
      self.relu1_2.add_module(str(x), features[x])

    for x in range(4, 7):
      self.relu2_1.add_module(str(x), features[x])

    for x in range(7, 9):
      self.relu2_2.add_module(str(x), features[x])

    for x in range(9, 12):
      self.relu3_1.add_module(str(x), features[x])

    for x in range(12, 14):
      self.relu3_2.add_module(str(x), features[x])

    for x in range(14, 16):
      self.relu3_3.add_module(str(x), features[x])

    for x in range(16, 18):
      self.relu4_1.add_module(str(x), features[x])

    for x in range(18, 21):
      self.relu4_2.add_module(str(x), features[x])

    for x in range(21, 23):
      self.relu4_3.add_module(str(x), features[x])

    for x in range(23, 26):
      self.relu5_1.add_module(str(x), features[x])

    for x in range(26, 28):
      self.relu5_2.add_module(str(x), features[x])

    for x in range(28, 30):
      self.relu5_3.add_module(str(x), features[x])

    # don't need the gradients, just want the features
    # for param in self.parameters():
    #    param.requires_grad = False

  def forward(self, x, layers=None, encode_only=False, resize=False):
    x = torch.cat([x, x, x], dim=1)
    relu1_1 = self.relu1_1(x)
    relu1_2 = self.relu1_2(relu1_1)

    relu2_1 = self.relu2_1(relu1_2)
    relu2_2 = self.relu2_2(relu2_1)

    relu3_1 = self.relu3_1(relu2_2)
    relu3_2 = self.relu3_2(relu3_1)
    relu3_3 = self.relu3_3(relu3_2)

    relu4_1 = self.relu4_1(relu3_3)
    relu4_2 = self.relu4_2(relu4_1)
    relu4_3 = self.relu4_3(relu4_2)

    relu5_1 = self.relu5_1(relu4_3)
    relu5_2 = self.relu5_2(relu5_1)
    relu5_3 = self.relu5_3(relu5_2)

    out = {
      'relu1_1': relu1_1,
      'relu1_2': relu1_2,

      'relu2_1': relu2_1,
      'relu2_2': relu2_2,

      'relu3_1': relu3_1,
      'relu3_2': relu3_2,
      'relu3_3': relu3_3,

      'relu4_1': relu4_1,
      'relu4_2': relu4_2,
      'relu4_3': relu4_3,

      'relu5_1': relu5_1,
      'relu5_2': relu5_2,
      'relu5_3': relu5_3,
    }
    if encode_only:
      if len(layers) > 0:
        feats = []
        for layer, key in enumerate(out):
          if layer in layers:
            feats.append(out[key])
        return feats
      else:
        return out['relu3_1']
    return out

class StyleLoss(nn.Module):
  r"""
  Perceptual loss, VGG-based
  https://arxiv.org/abs/1603.08155
  https://github.com/dxyang/StyleTransfer/blob/master/utils.py
  """

  def __init__(self):
    super(StyleLoss, self).__init__()
    self.add_module('vgg', VGG16())
    self.criterion = nn.L1Loss()

  def compute_gram(self, x):
    b, ch, h, w = x.size()
    f = x.view(b, ch, w * h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (b * h * w * ch)

    return G

  def __call__(self, x, y):
    # Compute features
    x_vgg, y_vgg = self.vgg(x), self.vgg(y)

    # Compute loss
    style_loss = 0.0
    style_loss += self.criterion(self.compute_gram(x_vgg['relu1_2']), self.compute_gram(y_vgg['relu1_2']))
    style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
    style_loss += self.criterion(self.compute_gram(x_vgg['relu3_3']), self.compute_gram(y_vgg['relu3_3']))
    style_loss += self.criterion(self.compute_gram(x_vgg['relu4_3']), self.compute_gram(y_vgg['relu4_3']))

    return style_loss

class PerceptualLoss(nn.Module):
  r"""
  Perceptual loss, VGG-based
  https://arxiv.org/abs/1603.08155
  https://github.com/dxyang/StyleTransfer/blob/master/utils.py
  """

  def __init__(self, weights=[0.0, 0.0, 1.0, 0.0, 0.0]):
    super(PerceptualLoss, self).__init__()
    self.add_module('vgg', VGG16())
    self.criterion = nn.L1Loss()
    self.weights = weights

  def __call__(self, x, y):
    # Compute features
    x_vgg, y_vgg = self.vgg(x), self.vgg(y)

    content_loss = 0.0
    content_loss += self.weights[0] * self.criterion(x_vgg['relu1_2'], y_vgg['relu1_2']) if self.weights[0] > 0 else 0
    content_loss += self.weights[1] * self.criterion(x_vgg['relu2_2'], y_vgg['relu2_2']) if self.weights[1] > 0 else 0
    content_loss += self.weights[2] * self.criterion(x_vgg['relu3_3'], y_vgg['relu3_3']) if self.weights[2] > 0 else 0
    content_loss += self.weights[3] * self.criterion(x_vgg['relu4_3'], y_vgg['relu4_3']) if self.weights[3] > 0 else 0
    content_loss += self.weights[4] * self.criterion(x_vgg['relu5_3'], y_vgg['relu5_3']) if self.weights[4] > 0 else 0

    return content_loss


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    """ 
    Returns a linear schedule of betas from start to end with an input timestep
    """
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device=device, noise=None):
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it
    """
    if noise is None:
      noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    # formula ->
    # mean = square root of alpha prod
    # std = square root of 1 - alpha prod
    mean = sqrt_alphas_cumprod_t.to(device) * x_0.to(device)
    var = sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
    # mean + variance
    return mean+var, noise.to(device)


# Define beta schedule
timesteps = configs.TIMESTEPS
betas = linear_beta_schedule(timesteps=timesteps)

# Pre-calculate different terms for closed form
# alpha = 1-beta
alphas = 1. - betas
# alphas_cumprod = prod(alpha)
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
target_real = Variable(torch.tensor(configs.BATCH_SIZE).fill_(1.0), requires_grad=False).to(device).float()
target_fake = Variable(torch.tensor(configs.BATCH_SIZE).fill_(0.0), requires_grad=False).to(device).float()
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()


distance_fn = nn.MSELoss()
criterionStyle = StyleLoss().to(device)
criterionFeature = PerceptualLoss().to(device)
criterion_GAN = nn.MSELoss()

LAMBDA = 10

def train_one_epoch(ep, models, optimizers, data_loaderA, data_loaderB, identity = False, iterations=100):
  # gen1 = generator for domain 1
  # gen2 = generator for domain 2
  # cyc12 = transform domain 1 to domain 2
  # cyc21 = transform domain 2 to domain 1
  genA, genB, cycAB, cycBA, netD_A, netD_B = models
  genA.train()
  genB.train()
  cycAB.train()
  cycBA.train()
  netD_A.train()
  netD_B.train()


  #define optimizers
  optim_gen, optim_cyc, optim_d = optimizers

  #tqdm loader
  pbar = tqdm(range(iterations), desc = f'Training Epoch {ep}')

  for i in pbar:

    #randonly sample images from 2 domain
    optim_gen.zero_grad()
    optim_cyc.zero_grad()
    optim_d.zero_grad()
    xA0 = next(iter(data_loaderA))[0].to(device)
    xB0 = next(iter(data_loaderB))[0].to(device)

    # begin diffusion
    tA = torch.randint(0, timesteps, (xA0.shape[0],), device=device).long()
    tB = torch.randint(0, timesteps, (xB0.shape[0],), device=device).long()
    noiseA = torch.randn_like(xA0, device=device)
    noiseB = torch.randn_like(xB0, device=device)
    xB0_fake = cycAB(xA0)
    pred_fake_B = netD_B(xB0_fake)
    xA0_fake = cycBA(xB0)
    pred_fake_A = netD_A(xA0_fake)


    #add noise
    xAtA, _ = forward_diffusion_sample(xA0, tA, device, noiseA)
    xBtB, _ = forward_diffusion_sample(xB0, tB, device, noiseB)
    xAtB_fake, _ = forward_diffusion_sample(xA0_fake, tB, device, noiseA)
    xBtA_fake, _ = forward_diffusion_sample(xB0_fake, tA, device, noiseB)


    #update diffusion weight
    predA = genA(torch.cat([xAtA, xBtA_fake.detach()], dim=1), tA)
    predB = genB(torch.cat([xBtB, xAtB_fake.detach()], dim=1), tB)
    diffusion_loss = distance_fn(predA, noiseA) + distance_fn(predB, noiseB)
    diffusion_loss.backward()
    optim_gen.step()

    #update cycle weight
    predA = genA(torch.cat([xAtA, xBtA_fake], dim=1), tA)
    predA_fake = genA(torch.cat([xAtB_fake, xBtB], dim=1), tB)
    predB = genB(torch.cat([xBtB, xAtB_fake], dim=1), tB)
    predB_fake = genB(torch.cat([xBtA_fake, xAtA], dim=1), tA)
    cyc_loss = distance_fn(predA, noiseA) + distance_fn(predB, noiseB) + distance_fn(predA_fake, noiseA) + distance_fn(predB_fake, noiseB)

    if identity:
      xA0_cyc = cycBA(xB0_fake)
      xB0_cyc = cycAB(xA0_fake)
      loss_style = criterionStyle(xA0, xA0_fake) + criterionStyle(xB0, xB0_fake)
      loss_feature = criterionFeature(xA0, xB0_fake) + criterionFeature(xB0, xA0_fake)
      loss_GAN_A2B = criterion_GAN(pred_fake_B, target_real)
      loss_GAN_B2A = criterion_GAN(pred_fake_A, target_real)
      cyc_loss += distance_fn(xA0_cyc, xA0)*LAMBDA + distance_fn(xB0_cyc, xB0)*LAMBDA + loss_feature*LAMBDA +loss_style*LAMBDA + loss_GAN_A2B + loss_GAN_B2A

    cyc_loss.backward()
    optim_cyc.step()

    ###### Discriminator A ######
    # Real loss
    pred_real = netD_A(xA0)
    loss_D_real = criterion_GAN(pred_real, target_real)

    # Fake loss
    xA0_fake = fake_A_buffer.push_and_pop(xA0_fake)
    pred_fake_A0 = netD_A(xA0_fake.detach())
    loss_D_fake = criterion_GAN(pred_fake_A0, target_fake)

    # Total loss
    loss_D_A = (loss_D_real + loss_D_fake) * 0.5
    ###################################

    ###### Discriminator B ######

    # Real loss
    pred_real = netD_B(xB0)
    loss_D_real = criterion_GAN(pred_real, target_real)

    # Fake loss
    xB0_fake = fake_B_buffer.push_and_pop(xB0_fake)
    pred_fake_B0 = netD_B(xB0_fake.detach())
    loss_D_fake = criterion_GAN(pred_fake_B0, target_fake)

    # Total loss
    loss_D_B = (loss_D_real + loss_D_fake) * 0.5

    loss_D = loss_D_A + loss_D_B
    ###################################

    loss_D.backward()
    optim_d.step()

    pbar.set_description(f"Epoch {ep} - Step {i+1}/{iterations} - Diff Loss = {round(diffusion_loss.item(),4)} - Cyc Loss = {round(cyc_loss.item(),4)}")
    
@torch.no_grad()
def translate_before_release(xA0, xBt, t, model):

  noiseA = torch.randn_like(xA0)
  noiseB = torch.randn_like(xBt)
  # if t.min() <= e:
  #   noiseA = noiseA*0
  #   noiseB = noiseB*0
  xAt, _ = forward_diffusion_sample(xA0, t, device, noiseA)

  betas_t = get_index_from_list(betas, t, xBt.shape)
  sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
      sqrt_one_minus_alphas_cumprod, t, xBt.shape
  )
  sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, xBt.shape)
  predicted_noise = model(torch.cat([xBt, xAt], dim=1), t)
  model_mean = sqrt_recip_alphas_t * (xBt - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
  posterior_variance_t = get_index_from_list(posterior_variance, t, xBt.shape)
  xBt_minus1 = model_mean + torch.sqrt(posterior_variance_t) * noiseB
  return xBt_minus1

@torch.no_grad()
def translate_after_release(xAt, xBt, t, model):
  '''
  removes noise from a noisy image at timestep t before release time
  unless we're sampling the final image - some noise is sampled to increase variance
  '''
  noiseA = torch.randn_like(xAt)
  noiseB = torch.randn_like(xBt)
  if t.min() <= e:
    noiseA = noiseA*0
    noiseB = noiseB*0

  betas_t = get_index_from_list(betas, t, xBt.shape)
  sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
      sqrt_one_minus_alphas_cumprod, t, xBt.shape
  )
  sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, xBt.shape)
  predicted_noise = model(torch.cat([xBt, xAt], dim=1), t)
  model_mean = sqrt_recip_alphas_t * (xBt - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
  posterior_variance_t = get_index_from_list(posterior_variance, t, xBt.shape)
  xBt_minus1 = model_mean + torch.sqrt(posterior_variance_t) * noiseB

  betas_t = get_index_from_list(betas, t, xAt.shape)
  sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
      sqrt_one_minus_alphas_cumprod, t, xAt.shape
  )
  sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, xAt.shape)
  predicted_noise = model(torch.cat([xAt, xBt], dim=1), t)
  model_mean = sqrt_recip_alphas_t * (xAt - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
  posterior_variance_t = get_index_from_list(posterior_variance, t, xAt.shape)
  xAt_minus1 = model_mean + torch.sqrt(posterior_variance_t) * noiseA

  return xAt_minus1, xBt_minus1

@torch.no_grad()
def translate(xA0, model, release_time=1):
  model.eval()
  b = xA0.shape[0]
  xBt = torch.randn(xA0.shape).to(device)
  noiseA = torch.randn_like(xA0)
  t = torch.full((b,), max(release_time,0), device=device, dtype=torch.long)
  xAt, _ = forward_diffusion_sample(xA0, t, device, noiseA)
  for i in tqdm(range(timesteps)):
    time = timesteps-i-1
    t = torch.full((b,), time, device=device, dtype=torch.long)
    if time > release_time:
      xBt = translate_before_release(xA0, xBt, t, model)
    else:
      xAt, xBt = translate_after_release(xAt, xBt, t, model)
  return xBt