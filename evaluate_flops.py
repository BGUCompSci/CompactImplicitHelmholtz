import time

import numpy as np
import torch

from fgmres import fgmres
from helmholtz import HelmholtzOperator, nn_precond
from multigrid import vcycle
from unet.implicit_unet import Encoder, ImplicitUNet
from unet.original_unet import FFSDNUnet, TFFKappa

device = torch.device('cuda')
# device = torch.device('cpu')

in_channels = 3
out_channels = 2
encoder_channels = [8, 16, 32, 64, 128]
tol = 1e-8
encoder_rounds = 1
solver_rounds = 6

model_type = 'original'

if model_type != 'vcycle':
    if model_type == 'implicit':
        encoder = Encoder(in_channels - 2, in_channels - 2, encoder_channels).to(device)
        solver = ImplicitUNet(in_channels, out_channels).to(device)
    elif model_type == 'explicit':
        encoder = Encoder(in_channels - 2, in_channels - 2, encoder_channels).to(device)
        solver = ImplicitUNet(in_channels, out_channels, implicit=False).to(device)
    elif model_type == 'original':
        encoder = TFFKappa(in_channels - 2).to(device)
        solver = FFSDNUnet(in_channels, out_channels).to(device)
    else:
        raise ValueError(f'Model type {model_type} not recognized')

    encoder.eval()
    solver.eval()

print(f'Using device {device}')
print(f'Using model {model_type}')


def sync_time():
    if device.type == 'cuda':
        torch.cuda.synchronize()

    return time.time()


with torch.no_grad():
    for size in [512, 1024, 2048, 4096]:
        kappa = torch.rand(size, size, device=device)
        gamma = torch.rand(size, size, device=device)

        if model_type != 'vcycle':
            encoding_durations = []

            for _ in range(encoder_rounds):
                start_time = sync_time()
                encoded_features = encoder(kappa.unsqueeze(0).unsqueeze(0))
                end_time = sync_time()

                encoding_duration = end_time - start_time
                encoding_durations.append(encoding_duration)

            print(f'Average encoding time for {size}: {np.mean(encoding_durations[1:])}, '
                  f'stddev: {np.std(encoding_durations[1:])}')

        op = HelmholtzOperator(kappa, 0.1, gamma, 0.1)

        if model_type == 'vcycle':
            precond = lambda x: vcycle(3, x.reshape(kappa.shape), op).flatten()
        else:
            precond = lambda x: nn_precond(op, lambda x: solver(x, encoded_features))

        durations = []
        for _ in range(solver_rounds):
            r = torch.rand(size, size, device=device, dtype=torch.cfloat)

            start_time = sync_time()
            res = fgmres(op, r.flatten(), max_restarts=3, max_iter=5, rel_tol=tol, precond=precond, flexible=True)
            end_time = sync_time()

            duration = (end_time - start_time) / res.num_iters
            durations.append(duration)

        print(f'Average time for size {size}: {np.mean(durations[1:])}, stddev: {np.std(durations[1:])}')
