import traceback
from pathlib import Path
from typing import NamedTuple, Union

import numpy as np
import torch
from hypy_utils import write
from hypy_utils.tqdm_utils import tmap, tq
from numpy.lib.scimath import sqrt as scisqrt
from scipy import linalg

from .emb_loader import EmbeddingLoader
from .model_loader import ModelLoader
from .utils import *


class FADInfResults(NamedTuple):
    score: float
    slope: float
    r2: float
    points: list[tuple[int, float]]

def qr_eigval(A:torch.Tensor, num_iterations=1000, tol=1e-8):
    A_k = A.clone()
    for _ in range(num_iterations):
        Q, R = torch.linalg.qr(A_k)
        A_k = R @ Q
        off_diagonal_norm = torch.norm(A_k - torch.diag(torch.diag(A_k)))
        if off_diagonal_norm < tol:
            break
    eigenvalues = torch.diag(A_k)
    return eigenvalues

def calc_frechet_distance_torch(x:torch.Tensor, y:torch.Tensor, device: str, precision=torch.float32, qr_iter=None, eps=1e-6) -> torch.Tensor:
    """FAD implementation in PyTorch.

    Args:
      x: The first set of embeddings of shape (n, embedding_dim).
      y: The second set of embeddings of shape (n, embedding_dim).
      precision: Type setting for matrix calculation precision.
      qr_iter: Number of iterations for QR decomposition.
               Defaults to eigendecomposition if not given.

    Returns:
      The FAD between x and y embedding sets.
    """
    
    # Apply precision setting
    if x.dtype is not precision or y.dtype is not precision:
        x = x.to(dtype=precision, device=device)
        y = y.to(dtype=precision, device=device)

    # Calculate mean vectors
    mu_x = torch.mean(x, axis=0)
    mu_y = torch.mean(y, axis=0)

    # Calculate mean distance term
    mu_diff = mu_x-mu_y
    diffnorm_sq = mu_diff@mu_diff
    
    # Calculate covariance matrices
    cov_x = torch.cov(x.T)
    cov_y = torch.cov(y.T)

    cov_prod = cov_x @ cov_y
    cov_prod.diagonal().add_(eps) # numerical stability

    if qr_iter:
        eig_val = qr_eigval(cov_prod, num_iterations=25, tol=eps)
        sqrt_eig_val = torch.sqrt(torch.clamp(eig_val, min=eps))
        tr_covmean = torch.sum(sqrt_eig_val)
    else:
        eig_val = torch.linalg.eigvals(cov_prod).real
        sqrt_eig_val = torch.sqrt(torch.clamp(eig_val, min=eps))
        tr_covmean = torch.sum(sqrt_eig_val)
        
    print("\n\n FAD cal: ", diffnorm_sq, torch.trace(cov_x), torch.trace(cov_y), 2*tr_covmean)
    return diffnorm_sq + torch.trace(cov_x) + torch.trace(cov_y) - 2*tr_covmean

def calc_embd_statistics(embd_lst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the mean and covariance matrix of a list of embeddings.
    """
    return np.mean(embd_lst, axis=0), np.cov(embd_lst, rowvar=False)


def calc_frechet_distance(mu1, cov1, mu2, cov2, eps=1e-6):
    """
    Adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    
    Numpy implementation of the Frechet Distance.
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
    -- cov1: The covariance matrix over activations for generated samples.
    -- cov2: The covariance matrix over activations, precalculated on an
            representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    cov1 = np.atleast_2d(cov1)
    cov2 = np.atleast_2d(cov2)

    assert mu1.shape == mu2.shape, \
        f'Training and test mean vectors have different lengths ({mu1.shape} vs {mu2.shape})'
    assert cov1.shape == cov2.shape, \
        f'Training and test covariances have different dimensions ({cov1.shape} vs {cov2.shape})'

    diff = mu1 - mu2

    # Product might be almost singular
    # NOTE: issues with sqrtm for newer scipy versions
    # using eigenvalue method as workaround
    covmean_sqrtm, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)
    
    # eigenvalue method
    D, V = linalg.eig(cov1.dot(cov2))
    covmean = (V * scisqrt(D)) @ linalg.inv(V)

    if not np.isfinite(covmean).all():
        offset = np.eye(cov1.shape[0]) * eps
        covmean = linalg.sqrtm((cov1 + offset).dot(cov2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    tr_covmean_sqrtm = np.trace(covmean_sqrtm)
    if np.iscomplexobj(tr_covmean_sqrtm):
        if np.abs(tr_covmean_sqrtm.imag) < 1e-3:
            tr_covmean_sqrtm = tr_covmean_sqrtm.real

    if not(np.iscomplexobj(tr_covmean_sqrtm)):
        delt = np.abs(tr_covmean - tr_covmean_sqrtm)
        if delt > 1e-3:
            print(f'WARNING: Detected high error in sqrtm calculation: {delt}')
            
    print("\n\n FAD cal: ", diff.dot(diff), np.trace(cov1), np.trace(cov2), 2*tr_covmean)
    return (diff.dot(diff) + np.trace(cov1)
            + np.trace(cov2) - 2 * tr_covmean)


class FrechetAudioDistance:
    def __init__(self, ml: ModelLoader, device: str, audio_load_worker: int = 8, logger = None, force_score_calc=False):
        self.ml = ml
        self.device = torch.device(device)
        self.emb_loader = EmbeddingLoader(ml, load_model=False)
        self.audio_load_worker = audio_load_worker
        self.logger = logger        
        self.force_score_calc = force_score_calc

        # Disable gradient calculation because we're not training
        torch.autograd.set_grad_enabled(False)
    
    def load_stats(self, path: PathLike):
        """
        Load embedding statistics from a directory.
        """
        if isinstance(path, str):
            # Check if it's a pre-computed statistic file
            bp = Path(__file__).parent / "fad_stats"
            stats = bp / (path.lower() + ".npz")
            if stats.exists():
                path = stats

        path = Path(path)

        # Check if path is a file
        if (not self.force_score_calc) and path.is_file():
            # Load it as a npz
            self.logger.info(f"Loading embedding statistics from {path}...")
            with np.load(path) as data:
                if f'{self.ml.name}.mu' not in data or f'{self.ml.name}.cov' not in data:
                    raise ValueError(f"FAD statistics file {path} doesn't contain data for model {self.ml.name}")
                return data[f'{self.ml.name}.mu'], data[f'{self.ml.name}.cov']
        cache_dir = path / "fad_stats" / self.ml.name
        emb_dir = path / "embeddings" / self.ml.name
        if (not self.force_score_calc) and cache_dir.exists():
            self.logger.info(f"Embedding statistics is already cached for {path}, loading...")
            mu = np.load(cache_dir / "mu.npy")
            cov = np.load(cache_dir / "cov.npy")
            return mu, cov
        
        if not path.is_dir():
            self.logger.error(f"The dataset you want to use ({path}) is not a directory nor a file.")
            exit(1)

        self.logger.info(f"Loading embedding files from {path}...")
        
        mu, cov = calculate_embd_statistics_online(list(emb_dir.glob("*.npy")))
        self.logger.info("> Embeddings statistics calculated.")

        # Save statistics
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / "mu.npy", mu)
        np.save(cache_dir / "cov.npy", cov)
        
        return mu, cov

    def score(self, baseline: PathLike, eval: PathLike):
        """
        Calculate a single FAD score between a background and an eval set.

        :param baseline: Baseline matrix or directory containing baseline audio files
        :param eval: Eval matrix or directory containing eval audio files
        """
        mu_bg, cov_bg = self.load_stats(baseline)
        mu_eval, cov_eval = self.load_stats(eval)

        return calc_frechet_distance(mu_bg, cov_bg, mu_eval, cov_eval)

    def score_inf(self, baseline: PathLike, eval_files: list[Path], steps: int = 25, min_n = 500, raw: bool = False):
        """
        Calculate FAD for different n (number of samples) and compute FAD-inf.

        :param baseline: Baseline matrix or directory containing baseline audio files
        :param eval_files: list of eval audio files
        :param steps: number of steps to use
        :param min_n: minimum n to use
        :param raw: return raw results in addition to FAD-inf
        """
        self.logger.info(f"Calculating FAD-inf for {self.ml.name}...")
        # 1. Load background embeddings
        mu_base, cov_base = self.load_stats(baseline)
        # If all of the embedding files end in .npy, we can load them directly
        if all([f.suffix == '.npy' for f in eval_files]):
            embeds = [np.load(f) for f in eval_files]
            embeds = np.concatenate(embeds, axis=0)
        else:
            embeds = self.emb_loader._load_embeddings(eval_files, concat=True)
        
        # Calculate maximum n
        max_n = len(embeds)

        # Generate list of ns to use
        ns = [int(n) for n in np.linspace(min_n, max_n, steps)]
        
        results = []
        for n in tq(ns, desc="Calculating FAD-inf"):
            # Select n feature frames randomly (with replacement)
            indices = np.random.choice(embeds.shape[0], size=n, replace=True)
            embds_eval = embeds[indices]
            
            mu_eval, cov_eval = calc_embd_statistics(embds_eval)
            fad_score = calc_frechet_distance(mu_base, cov_base, mu_eval, cov_eval)

            # Add to results
            results.append([n, fad_score])

        # Compute FAD-inf based on linear regression of 1/n
        ys = np.array(results)
        xs = 1 / np.array(ns)
        slope, intercept = np.polyfit(xs, ys[:, 1], 1)

        # Compute R^2
        r2 = 1 - np.sum((ys[:, 1] - (slope * xs + intercept)) ** 2) / np.sum((ys[:, 1] - np.mean(ys[:, 1])) ** 2)

        # Since intercept is the FAD-inf, we can just return it
        return FADInfResults(score=intercept, slope=slope, r2=r2, points=results)
    
    def score_individual(self, baseline: PathLike, eval_dir: PathLike, csv_name: Union[Path, str]) -> Path:
        """
        Calculate the FAD score for each individual file in eval_dir and write the results to a csv file.

        :param baseline: Baseline matrix or directory containing baseline audio files
        :param eval_dir: Directory containing eval audio files
        :param csv_name: Name of the csv file to write the results to
        :return: Path to the csv file
        """
        csv = Path(csv_name)
        if isinstance(csv_name, str):
            csv = Path('data') / f'fad-individual' / self.ml.name / csv_name
        if csv.exists():
            self.logger.info(f"CSV file {csv} already exists, exiting...")
            return csv

        # 1. Load background embeddings
        mu, cov = self.load_stats(baseline)

        # 2. Define helper function for calculating z score
        def _find_z_helper(f):
            try:
                # Calculate FAD for individual songs
                embd = self.emb_loader.read_embedding_file(f)
                mu_eval, cov_eval = calc_embd_statistics(embd)
                return calc_frechet_distance(mu, cov, mu_eval, cov_eval)

            except Exception as e:
                traceback.print_exc()
                self.logger.error(f"An error occurred calculating individual FAD using model {self.ml.name} on file {f}")
                self.logger.error(e)

        # 3. Calculate z score for each eval file
        _files = list(Path(eval_dir).glob("*.*"))
        scores = tmap(_find_z_helper, _files, desc=f"Calculating scores", max_workers=self.audio_load_worker)

        # 4. Write the sorted z scores to csv
        pairs = list(zip(_files, scores))
        pairs = [p for p in pairs if p[1] is not None]
        pairs = sorted(pairs, key=lambda x: np.abs(x[1]))
        write(csv, "\n".join([",".join([str(x).replace(',', '_') for x in row]) for row in pairs]))

        return csv

