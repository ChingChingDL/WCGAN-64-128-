#################################################
import torch
import os
import cv2
import time
import warnings
import parameters as p
from model import Discriminator,Generator
from utils import load_model,load_optimizer
from utils import sample_and_exhibit
from utils import save_model
from utils import save_optimizer
from utils import gradient_penalty
from dataset import get_dataloader

#################################################
warnings.filterwarnings("ignore", category=UserWarning)
def train(generator: Generator, critic: Discriminator,
          opt_critic: torch.optim.Optimizer,
          opt_gen: torch.optim.Optimizer,
          dataloader, device,
          epochs: int = p.EPOCHS,
          save_on_step: int = p.SAVE_ON_STEP,
          sample_on_step: int = p.SAMPLE_ON_STEP,
          verbose_on_step:int = p.VERBOSE_ON_STEP):

    if os.path.exists(p.MODEL_CHECKPOINT_PATH) and load_model(critic, generator,strict=False):
        print("################################## models loaded ##################################\n")
    else:
        pass
        # initialize_weights(critic)
        # initialize_weights(generator)
    if os.path.exists(p.OPTIMIZER_CHECKPOINT_PATH) and load_optimizer(opt_critic, opt_gen):
        print("################################## optimizers loaded ##################################\n")
    opt_critic.param_groups[0]['initial_lr'] = p.LEARNING_RATE
    opt_gen.param_groups[0]['initial_lr'] = p.LEARNING_RATE
    cv2.namedWindow('Sample', cv2.WINDOW_AUTOSIZE)
    # cv2.namedWindow('reals',cv2.WINDOW_AUTOSIZE)
    generator = generator.to(device)
    critic = critic.to(device)
   
    print("################################## infomation ##################################")
    print(f"Batchs : {len(dataloader)}")
    print(f"Batch size : {p.BATCH_SIZE}")
    print(f"current device : {device}")
    print(f"")
    print('################################## ##################################\n')
    if len(dataloader) < sample_on_step \
    or len(dataloader) < save_on_step \
    or len(dataloader) < verbose_on_step:
        print(f"unreasonable operation_on_step:\n total steps: {len(dataloader)} \n sample_on_step: {sample_on_step} \n verbose_on_step: {verbose_on_step} \n save_on_step: {save_on_step}")
        exit()
    # one = torch.tensor(1, dtype=torch.float,device=device)
    # mone = (one * -1).to(device)

    # 41 20230417
    # 41- 91 20230418
    # 91 - 101 20230419
    # 102 - 116
    for epoch in range(0,epochs):
        start_time = time.time()
        print(f"epoch : {epoch} of {epochs}")
        loss_accumulator = {'critic': 0., 'generator': 0.,'gp':0.,'w_distance':0.}
        
        
        for batch_idx,(x, _) in enumerate(dataloader):
            real = x.to(device)

            
            for param in critic.parameters():
                param.requires_grad = True

            for __ in range(p.CRITIC_ITERATIONS):
                critic.zero_grad()

                critic_real = critic(real).mean()
            
                z = torch.randn([real.size(0),*p.LATENT_SINGLE_SHAPE], device=device)
                fake = generator(z)
                critic_fake = critic(fake).mean()
        
                gp = gradient_penalty(critic,real,fake,device=device)  
                loss_critic = critic_fake - critic_real + gp
                loss_critic.backward()

                opt_critic.step()
                  
            for param in critic.parameters():
                param.requires_grad = False

            # train generator
            for __ in range(p.GENERATOR_ITERATIONS):
                generator.zero_grad()

                z = torch.randn([real.size(0),*p.LATENT_SINGLE_SHAPE], device=device)
                fake = generator(z)        
                output = critic(fake).mean()
                loss_gen = - output
                loss_gen.backward()
                opt_gen.step()

            
            loss_accumulator['generator'] += loss_gen.cpu().detach().item()
            loss_accumulator['critic'] += loss_critic.cpu().detach().item()
            loss_accumulator['gp'] += gp.cpu().detach().item()
            loss_accumulator['w_distance'] += (critic_real - critic_fake).cpu().detach().item()

            if batch_idx % verbose_on_step == 0 :
                critic_loss = loss_accumulator['critic'] / verbose_on_step
                generator_loss = loss_accumulator["generator"] / verbose_on_step
                avg_gp = loss_accumulator["gp"] / verbose_on_step
                avg_w_d = loss_accumulator["w_distance"] / verbose_on_step
                end_time = time.time()
                print(
            f"[critic loss : {critic_loss:<12}] [ generator loss : {generator_loss:>12}]\n [EM_distance : {avg_w_d}] [gp : {avg_gp}]({batch_idx+1}/{epoch+1})")
                print(f'time cost : {end_time - start_time }') 
                
                loss_accumulator = {'critic': 0., 'generator': 0.,'gp':0.,'w_distance':0.}
                start_time = time.time()

            if batch_idx % sample_on_step == 0:
                sample_and_exhibit(winname='Sample',winname2='reals',generator=generator,device=device)
                # sample_and_exhibit('',winname2='reals',device=device,samples=real)
            if batch_idx != 0 and batch_idx %  save_on_step == 0:
                 
                if not os.path.exists(p._SAVE_PATH):
                    os.mkdir(p._SAVE_PATH)

                if save_model(D=critic, G=generator):
                    print(
                        "################################## models saved ##################################")
                else:
                    print("Fail to save models")
                if save_optimizer(opt_critic=opt_critic, opt_gen=opt_gen):
                    print(
                        "################################## optimizers saved ##################################")
                else:
                    print("Fail to save optimizers")
                    
   
        ##########################################################

        

        ##########################################################


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    opt_critic = torch.optim.Adam([{'initial_lr': p.LEARNING_RATE,'params': discriminator.parameters(), }], 
                                  lr=p.LEARNING_RATE,betas=p.BETAS,)
    opt_gen = torch.optim.Adam([{'initial_lr': p.LEARNING_RATE,'params': generator.parameters()} ], 
                               lr=p.LEARNING_RATE,betas=p.BETAS,
                               )
    dataloader = get_dataloader(presistent_workers=True)
    train(generator=generator, critic=discriminator,
          opt_critic=opt_critic, opt_gen=opt_gen,
          dataloader=dataloader, device=device)
