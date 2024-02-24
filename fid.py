import os
import csv
import pickle
import random
import numpy as np
from glob import glob
from PIL import Image
from scipy import linalg
from collections import defaultdict
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


import torch
import torchvision.transforms as TF
from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from classifier.densenet import DenseNet
from inception_fid import InceptionV3
import torchxrayvision as xrv
import skimage

from vae import VQGanVAE
import albumentations.pytorch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--gt_path', type=str, default=None, help='path to the original image path')
parser.add_argument('--gen_path', type=str, default=None, help='path to the generated image path')

parser.add_argument('--codebook_indices', type=str, default='mimiccxr_vqgan/last.ckpt', help='path to the codebook indices path')
parser.add_argument('--vagan', type=str, default='mimiccxr_vqgan/last.ckpt', help='path to the vqgan model path')
parser.add_argument('--vagan_config', type=str, default='mimiccxr_vqgan/last.ckpt', help='path to the vqgam config path')

parser.add_argument('--dims', type=int, default=1024, help='Batch size to use')
parser.add_argument('--batch-size', type=int, default=50, help='Batch size to use')
parser.add_argument('--num-workers', type=int, help=('Number of processes to use for data loading. Defaults to `min(8, num_cpus)`'))
parser.add_argument('--device', type=str, default=None, help='Device to use. Like cuda, cuda:0 or cpu')


IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, args, files, transforms=None):
        self.files = files
        self.transforms = transforms

        with open(args.codebook_indices, 'rb') as f:
            self.indices_dict = pickle.load(f)

        vqgan_model_path = args.vqgan
        vqgan_config_path = args.vqgan_config
        self.vae = VQGanVAE(vqgan_model_path, vqgan_config_path)  # .to(device)

        IMG_SIZE = self.vae.image_size

        rescaler = albumentations.SmallestMaxSize(max_size=IMG_SIZE)
        cropper = albumentations.CenterCrop(height=IMG_SIZE, width=IMG_SIZE)
        # totensor = albumentations.pytorch.transforms.ToTensorV2()
        self.image_transform = albumentations.Compose([
            rescaler,
            cropper,
            # totensor
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]

        img = skimage.io.imread(path)
        img = xrv.datasets.normalize(img, 255)  # convert 8-bit image to [-1024, 1024] range
        if len(img.shape) == 3:
            img = img.mean(2)[None, ...]  # Make single color channel
        else:
            img = img[None, ...]

        transform = TF.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])

        img = transform(img)
        img = torch.from_numpy(img)

        return img


def get_activations(args, files, model, batch_size=50, dims=2048, device='cpu', num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    print('total len of files:', len(files))
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)


    dataset = ImagePathDataset(args, files, transforms=TF.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(files), dims))

    start_idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device)

            with torch.no_grad():
                    pred = model.features(batch)

            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()

            pred_arr[start_idx:start_idx + pred.shape[0]] = pred

            start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(args, files, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(args, files, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(args, path, model, batch_size, dims, device, num_workers):
    m, s = calculate_activation_statistics(args, path, model, batch_size, dims, device, num_workers)
    return m, s


def calculate_fid_given_paths(args, gt_path, gen_path, batch_size, device, dims, num_workers=1):
    """Calculates the FID of two paths"""
    model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device)
    m1, s1 = compute_statistics_of_path(args, gt_path, model, batch_size, dims, device, num_workers)
    m2, s2 = compute_statistics_of_path(args, gen_path, model, batch_size, dims, device, num_workers)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def main():
    args = parser.parse_args()
    set_seed(123)
    gt_files = glob(args.gt_path)
    gen_files = glob(args.gt_path)
    fid_value = calculate_fid_given_paths(args, gt_files, gen_files, args.batch_size, device, args.dims, num_workers)
    print('FID: ', round(fid_value, 3))

if __name__ == '__main__':
    main()