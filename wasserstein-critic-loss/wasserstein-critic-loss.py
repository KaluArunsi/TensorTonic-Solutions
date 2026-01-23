import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein Critic Loss for WGAN.
    """
    real_scores, fake_scores = np.asanyarray(real_scores), np.asanyarray(fake_scores)

    mean_real, mean_fake = np.mean(real_scores), np.mean(fake_scores)

    L = mean_fake - mean_real

    return L