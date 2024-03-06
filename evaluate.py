import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data import KappaDataGenerator, MinMaxScalerVectorized, OpenFWIDataset
from fgmres import fgmres
from helmholtz import HelmholtzOperator, absorbing_layer, nn_precond
from multigrid import vcycle
from unet.implicit_unet import EncoderSolver

DATASET = "stl10"
TOP_TRAIN_SIZE = 256

evaluation_title = (f"Evaluating V-cycle trained up to {TOP_TRAIN_SIZE} "
                    f"using dataset {DATASET}, but evaluated on OpenFWI")

model_paths = ['implicit/model.ckpt', 'explicit/model.ckpt']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = [EncoderSolver.load_from_checkpoint(path, map_location=device, in_channels=3, out_channels=2, small=True)
          .to(device).eval() for path in model_paths]


def single_source(op, model, iterations=50):
    kappa = op.kappa
    height, width = [*kappa.shape]
    tol = 1e-7

    b = torch.zeros_like(kappa, dtype=torch.cfloat)
    b[height // 2, width // 2] = 1 / ((op.h) ** 2)

    precond = lambda x: vcycle(3, x.reshape(kappa.shape), op).flatten()
    x0 = fgmres(op, b.flatten(), rel_tol=tol, max_iter=3, max_restarts=1, precond=precond,
                flexible=True).solution

    precond = lambda x: nn_precond(op, model, x)
    x_sol = fgmres(op, b.flatten(), max_restarts=5, rel_tol=tol, max_iter=iterations,
                   precond=precond, x0=x0, flexible=True, save_intermediate=True)

    residuals = x_sol.residual_norms
    return x_sol.solution, torch.tensor(residuals)


with torch.no_grad():
    grids = [128, 256, 512, 1024, 2048, 4096]
    frequencies = [10, 20, 40, 80, 160, 320]
    gamma_value = 0.01
    for i, grid in enumerate(grids):
        height = width = grid
        f = frequencies[i]
        omega = 2 * torch.pi * f
        h = 2.0 / (height + width)
        gamma = gamma_value * omega * torch.ones(height, width, device=device)

        # Old Kappa, lets use STL10
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((height, width), antialias=True),
                                        MinMaxScalerVectorized(feature_range=(0.25, 1))])
        dataset = OpenFWIDataset("/home/barlere/helunet/data/stylea", 500, transform)
        data_loader = DataLoader(dataset, 1, True, num_workers=4, pin_memory=True)
        kappa_iterator = iter(data_loader)

        # STL10 Kappa
        datasetSTL = KappaDataGenerator(height, width)
        datasetSTL.load_data()

        implicit_iterations = []
        explicit_iterations = []
        implicit_times = []
        explicit_times = []
        kappa = torch.zeros_like(gamma)
        gamma = absorbing_layer(gamma, [16, 16], omega)

        op = HelmholtzOperator(kappa, omega, gamma, h)

        for _ in range(100):
            # This is for OpenFWI
            kappa = next(kappa_iterator)[0].reshape(height, width).to(device, torch.float32)
            # This is for STL10
            # kappa = datasetSTL.generate_kappa().reshape(height, width).to(device, torch.float32)
            op.kappa = kappa

            # Run experiment for single-source
            torch.cuda.synchronize()
            start_time = time.time()
            _, unet_residuals = single_source(op, models[0])
            torch.cuda.synchronize()
            end_time = time.time()

            implicit_iterations.append(len(unet_residuals))
            implicit_times.append(end_time - start_time)

            torch.cuda.synchronize()
            start_time = time.time()

            _, unet_residuals = single_source(op, models[1])
            torch.cuda.synchronize()
            end_time = time.time()

            explicit_iterations.append(len(unet_residuals))
            explicit_times.append(end_time - start_time)

        print(
            f"Grid {grid} average for implicit {np.mean(implicit_iterations)} and Explicit {np.mean(explicit_iterations)}")
        print(f"Grid {grid} time average for implicit {np.mean(implicit_times)} and Explicit {np.mean(explicit_times)}")
