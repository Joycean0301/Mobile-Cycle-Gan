from torchvision.models import inception_v3

import torch
import numpy as np
import scipy.linalg as splinalg


def calculate_fid_score(real_images, generated_images, model, device, batch_size):
    """
    Computes Fréchet Inception Distance (FID) between real and generated images
    using a pretrained inception network.

    Arguments:
    real_images -- a PyTorch tensor of real images
    generated_images -- a PyTorch tensor of generated images
    model -- a pretrained inception network
    device -- the device on which the computations should be run
    batch_size -- the batch size to use when computing activations
    dims -- the number of dimensions in the activations of the inception network

    Returns:
    fid -- the Fréchet Inception Distance between the real and generated images
    """
    model.to(device)
    model.eval()

    # Compute activations for real images
    with torch.no_grad():
        real_images = real_images.to(device)
        activations_real = []
        for i in range(0, len(real_images), batch_size):
            activations_real.append(model(real_images[i:i+batch_size]).detach().cpu().numpy())
        activations_real = np.concatenate(activations_real, axis=0)
        mean_real, cov_real = np.mean(activations_real, axis=0), np.cov(activations_real, rowvar=False)

    # Compute activations for generated images
    with torch.no_grad():
        generated_images = generated_images.to(device)
        activations_generated = []
        for i in range(0, len(generated_images), batch_size):
            activations_generated.append(model(generated_images[i:i+batch_size]).detach().cpu().numpy())
        activations_generated = np.concatenate(activations_generated, axis=0)
        mean_generated, cov_generated = np.mean(activations_generated, axis=0), np.cov(activations_generated, rowvar=False)

    # Compute FID
    mean_diff = mean_real - mean_generated
    cov_mean = (cov_real + cov_generated) / 2
    fid = np.sqrt(np.sum(mean_diff**2) + cov_real + cov_generated - 2 * np.sqrt(cov_mean))

    return fid

inception_v3_model = inception_v3(pretrained=True, transform_input=False)