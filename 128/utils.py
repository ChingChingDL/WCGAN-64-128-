import cv2
import torch
from torch import nn
import einops
import parameters as p
from model import Generator128, Discriminator128

def load_model(critic: Discriminator128, G: Generator128,strict=False):
    try:
        model_states = torch.load(p.MODEL_CHECKPOINT_PATH)
        critic.load_state_dict(model_states['critic'],strict=strict)
        G.load_state_dict(model_states['gen'],strict=strict)
    except Exception:
        return False

    return True

def save_model(D: Discriminator128, G: Generator128):

    try:
        ctritic_state = D.state_dict()
        gen_state = G.state_dict()

        torch.save({'critic':ctritic_state,'gen':gen_state}, p.MODEL_CHECKPOINT_PATH)
    except Exception:
        return False
    return True

def load_optimizer(opt_critic:torch.optim.Optimizer, opt_gen:torch.optim.Optimizer):
    try:
        opt_state = torch.load(p.OPTIMIZER_CHECKPOINT_PATH)
        
        opt_critic.load_state_dict(opt_state['opt_critic'])
        opt_gen.load_state_dict(opt_state['opt_gen'])
        
    except Exception:
        return False

    return True


def save_optimizer(opt_critic:torch.optim.Optimizer, opt_gen:torch.optim.Optimizer):
    try:
        opt_critic_state = opt_critic.state_dict()
        opt_gen_state = opt_gen.state_dict()
        torch.save({'opt_critic':opt_critic_state,'opt_gen':opt_gen_state}, p.OPTIMIZER_CHECKPOINT_PATH)
    except Exception:
        return False
    return True


def gradient_penalty(critic,real,fake,device='cpu'):
    BATCH_SIZE,C,H,W = real.shape 
    epsilon = torch.rand([BATCH_SIZE,1,1,1]).repeat(1,C,H,W).to(device)
    interpolation_images = real * epsilon + fake * (1-epsilon)
    
    #
    mixed_scores = critic(interpolation_images)

    gradient = torch.autograd.grad(
        inputs=interpolation_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0],-1)
    gradient_norm = gradient.norm(2,dim=1)

    return torch.mean((gradient_norm - 1) ** 2) * p.LAMBDA_GP


@torch.no_grad()
def sample_and_exhibit(winname,winname2,generator=None,device=None,samples=None):

    if generator is not None:
        # generator.eval()
        z = torch.randn([16,*p.LATENT_SINGLE_SHAPE], device=device)
        sample = generator(z).cpu()
        sample = einops.rearrange(
            sample, '(b_x b_y) c h w -> (b_y h) (b_x w) c', b_x=4).numpy()
        # sample = make_grid(sample[:],nrow=2)
        sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
        cv2.imshow(winname, sample)
        cv2.waitKey(10)
    else :
        if samples is not None:
            sample = samples.chunk(int(p.BATCH_SIZE/4),0)[0]
            sample = einops.rearrange(
            sample, '(b_x b_y) c h w -> (b_y h) (b_x w) c', b_x=8).cpu().numpy()
            # sample = make_grid(sample[:],nrow=2)
            sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
            cv2.imshow(winname2, sample)
            cv2.waitKey(20)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)