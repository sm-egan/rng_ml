import torch
import numpy as np
import matplotlib.pyplot as plt
from aes_prng import BatchedAESRandomGenerator,HWAccelBatchedAESRandomGenerator, HWAccelAESGenerator
from scipy import stats

def verify_distributions(n_samples: int = 4_000_000, bins: int = 100):
    """
    Generate and plot distributions from BatchedAESRandomGenerator.
    Compares against expected theoretical distributions.
    """
    # Initialize generators
    aes_gen = BatchedAESRandomGenerator()
    #aes_gen = HWAccelAESGenerator()
    torch_gen = torch.Generator()
    
    # Generate samples
    aes_uniform = aes_gen.rand(n_samples).numpy()
    aes_normal = aes_gen.randn(n_samples).numpy()
    # aes_uniform = torch.rand(n_samples, generator=aes_gen).numpy()
    # aes_normal = torch.randn(n_samples, generator=aes_gen).numpy()
    torch_uniform = torch.rand(n_samples, generator=torch_gen).numpy()
    torch_normal = torch.randn(n_samples, generator=torch_gen).numpy()
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot uniform distributions
    ax1.hist(aes_uniform, bins=bins, density=True, alpha=0.6, label='AES Uniform')
    ax1.hist(torch_uniform, bins=bins, density=True, alpha=0.6, label='PyTorch Uniform')
    #ax1.set_yscale('log')
    ax1.plot([0, 1], [1, 1], 'r--', label='Expected Uniform')
    ax1.set_title('Uniform Distribution Comparison')
    ax1.legend()
    
    # Plot normal distributions
    ax2.hist(aes_normal, bins=bins, density=True, alpha=0.6, label='AES Normal')
    ax2.hist(torch_normal, bins=bins, density=True, alpha=0.6, label='PyTorch Normal')
    x = np.linspace(-4, 4, 100)
    ax2.plot(x, stats.norm.pdf(x), 'r--', label='Expected Normal')
    ax2.set_title('Normal Distribution Comparison')
    ax2.legend()
    
    # Q-Q plots
    stats.probplot(aes_uniform, dist='uniform', plot=ax3)
    ax3.set_title('Q-Q Plot: AES Uniform vs Expected Uniform')
    
    stats.probplot(aes_normal, dist='norm', plot=ax4)
    ax4.set_title('Q-Q Plot: AES Normal vs Expected Normal')
    
    # Add statistical tests
    ks_uniform = stats.kstest(aes_uniform, 'uniform')
    ks_normal = stats.kstest(aes_normal, 'norm')
    ks_uniform_torch = stats.kstest(torch_uniform, 'uniform')
    ks_normal_torch = stats.kstest(torch_normal, 'norm')
    
    plt.figtext(0.3, 0.08, f'Kolmogorov-Smirnov test results:\n'
                f'AES Uniform: p={ks_uniform.pvalue:.4f}\n'
                f'AES Normal: p={ks_normal.pvalue:.4f}\n'
                f'PyTorch Uniform: p={ks_uniform_torch.pvalue:.4f}\n'
                f'PyTorch Normal: p={ks_normal_torch.pvalue:.4f}',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    verify_distributions()
