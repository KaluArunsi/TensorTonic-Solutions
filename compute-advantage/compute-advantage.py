import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """
    V, rewards = np.asanyarray(V), np.asanyarray(rewards)

    g_t = np.zeros_like(rewards, dtype=float)
    g_value = 0

    for r in reversed(range(len(rewards))):
        # apparently you can recursively just do
        # G_current = r + (gamma * G_previous)
        g_value = rewards[r] + (gamma * g_value)
        g_t[r] = g_value
        
    advantages = g_t - V

    return advantages

