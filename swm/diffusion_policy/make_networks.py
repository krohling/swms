import torch
import torch.nn as nn
from diffusers import DDPMScheduler, EMAModel, get_scheduler
from .networks import ConditionalUnet1D, get_resnet, replace_bn_with_gn
from .configs import DiffusionModelRunConfig
from swm.utils.dataset import DiffusionDataset

def instantiate_model_artifacts(cfg: DiffusionModelRunConfig, model_only: bool = False, device=None):
    '''
    Instantiate the model and the training objects.
    If model only, returns network and scheduler and device only
    If not model only, returns network, ema, noise scheduler, optimizer, lr_scheduler, dataloader, stats, device
    '''
    if device is None:
        device = torch.device('cuda')
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.num_diffusion_iters,
        # the choice of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )

    # ResNet18 has output dim of 512
    obs_dim = 0
    if cfg.with_image:
        obs_dim += 512 * cfg.num_cameras
    if cfg.with_state:
        obs_dim += cfg.state_len

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=cfg.action_dim,
        global_cond_dim=obs_dim * cfg.obs_horizon
    )

    # the final arch has 2 parts
    nets_dict = {
        'noise_pred_net': noise_pred_net
    }
    if cfg.with_image:
        vision_encoder = get_resnet('resnet18')
        vision_encoder = replace_bn_with_gn(vision_encoder)
        nets_dict["vision_encoder"] = vision_encoder

    nets = nn.ModuleDict(nets_dict)
    # device transfer
    _ = nets.to(device)

    if model_only:
        return nets, noise_scheduler, device

    dataset = DiffusionDataset(cfg.dataset.data_folder_path, cfg.pred_horizon, cfg.obs_horizon, pad_length=cfg.dataset.padding)
    stats = dataset.stats
    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process after each epoch
        persistent_workers=True
    )

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=nets.parameters(),
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parameter are not optimized
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=(len(dataloader) * cfg.num_epochs) // 10,
        num_training_steps=len(dataloader) * cfg.num_epochs
    )
    return nets, ema, noise_scheduler, optimizer, lr_scheduler, dataloader, stats, device
