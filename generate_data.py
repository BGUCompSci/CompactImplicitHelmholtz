import argparse
import random

import torch

from data import KappaDataGenerator
from fgmres import fgmres
from helmholtz import HelmholtzOperator, absorbing_layer
from multigrid import vcycle


def generate_data(path: str, samples: int, normalize: bool = False):
    height = width = 256
    f = 20
    gamma_value = 0.05
    omega = 2 * torch.pi * f
    h = 2 / (height + width)

    # dataset = OpenFWIDataset(kappa_dataset_path, 500, transform)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    dataset = KappaDataGenerator(height, width)
    dataset.load_data()

    for i in range(samples):
        gamma = gamma_value * omega * torch.ones(height, width)
        gamma = absorbing_layer(gamma, [16, 16], omega)
        kappa = dataset.generate_kappa().reshape(height, width)

        op = HelmholtzOperator(kappa, omega, gamma, h)

        # Generate true solution and RHS
        x_sol = torch.randn(height, width, dtype=torch.cfloat)
        b = op(x_sol).reshape_as(kappa)

        # Solve to get x_k
        iterations = random.randint(2, 20)
        precond = lambda x: vcycle(3, x.reshape(kappa.shape), op).flatten()
        x_k = fgmres(op, b.flatten(), iterations, 1e-20, 1, precond=precond, flexible=True).solution.reshape(
            height, width)

        # Calculate residual and get the error
        error = (x_sol - x_k)
        residual = (b - op(x_k)).reshape(height, width) * (h ** 2)

        if normalize:
            residual /= torch.norm(residual)
            error /= torch.norm(residual)

        image = torch.stack((residual.real, residual.imag, kappa))
        error = torch.stack((error.real, error.imag))
        # Save in the data folder
        torch.save(image, f"{path}/dataimage{i}.pt")
        torch.save(error, f"{path}/error{i}.pt")


def main():
    arg_parser = argparse.ArgumentParser(description="Generates data for the UNet")
    arg_parser.add_argument('-s', "--samples", help="The number of samples to generate", type=int, required=True)
    arg_parser.add_argument('-p', '--path', help="Specifiy the path of the folder to save the samples", type=str)
    arg_parser.add_argument(
        '-n', '--norm', help="Specifiy if the networks input and output should be normelized",
        action='store_true')

    args = arg_parser.parse_args()
    generate_data(args.path, args.samples, args.norm)


if __name__ == "__main__":
    main()
