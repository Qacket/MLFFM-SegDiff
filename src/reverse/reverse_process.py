import torch
from tqdm import tqdm
from models import DermoSegDiff, MLFFMSegDiff



@torch.no_grad()
def p_sample(forward_schedule, model, images, x, t, t_index, clip_x0=False):
    fs = forward_schedule

    exs = fs.extract(t, x.shape)
    betas_t = exs["betas"]
    sqrt_one_minus_alphas_cumprod_t = exs["sqrt_one_minus_alphas_cumprod"]
    sqrt_recip_alphas_t = exs["sqrt_recip_alphas"]

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean

    if isinstance(model, (DermoSegDiff, MLFFMSegDiff)):
        predicted_noise = model(x, images, t)
    else:
        NotImplementedError('given model is unknown!')

    if clip_x0:
        # posterior mean q(x_{t-1} | x_t, x0) with the predicted x0 clipped to the mask range [-1, 1]
        alphas_cumprod_t = exs["alphas_cumprod"]
        alphas_cumprod_prev_t = exs["alphas_cumprod_prev"]
        x0 = (x - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / torch.sqrt(alphas_cumprod_t)
        x0 = x0.clamp(-1.0, 1.0)
        model_mean = (
            betas_t * torch.sqrt(alphas_cumprod_prev_t) / (1.0 - alphas_cumprod_t) * x0
            + (1.0 - alphas_cumprod_prev_t) * torch.sqrt(exs["alphas"]) / (1.0 - alphas_cumprod_t) * x
        )
    else:
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = exs["posterior_variance"]
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(forward_schedule, model, images, out_channels, desc="", clip_x0=False):
    timesteps = forward_schedule.timesteps
    device = next(model.parameters()).device

    shape = (images.shape[0], out_channels, images.shape[2], images.shape[3])
    b = shape[0]

    # start from pure noise (for each example in the batch)
    msk = torch.randn(shape, device=device)
    msks = []

    desc = f"{desc} - sampling loop time step" if desc else "sampling loop time step"
    for i in tqdm(
        reversed(range(0, timesteps)), desc=desc, total=timesteps, leave=False
    ):
        msk = p_sample(
            forward_schedule,
            model,
            images,
            msk,
            torch.full((b,), i, device=device, dtype=torch.long),
            i,
            clip_x0=clip_x0,
        )
        msks.append(msk.detach().cpu())
    return msks


@torch.no_grad()
def sample(forward_schedule, model, images, out_channels=2, desc=None, clip_x0=False):
    return p_sample_loop(forward_schedule, model, images, out_channels, desc, clip_x0=clip_x0)



def reverse_by_epsilon(forward_process, predicted_noise, x, t):
    fs = forward_process.forward_schedule
    exs = fs.extract(t, x.shape)
    
    betas_t = exs["betas"]
    sqrt_one_minus_alphas_cumprod_t = exs["sqrt_one_minus_alphas_cumprod"]
    sqrt_recip_alphas_t = exs["sqrt_recip_alphas"]
    posterior_variance_t = exs["posterior_variance"]

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
    )
    
    noise = torch.randn_like(x) if t[0].item() > 0 else 0
    # Algorithm 2 line 4:
    res = model_mean + torch.sqrt(posterior_variance_t) * noise

    return res
