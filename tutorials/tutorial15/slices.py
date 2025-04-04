import torch
import numpy as np
import matplotlib.pyplot as plt

# Load new PINN output
output_test = torch.load("pinn_output_post_tuning.pt", weights_only=False)
u_pinn_new = output_test.extract(["u"]).detach().numpy().reshape(50, 50).T  # Reshape to (time, space)

# Load old PINN & FDM output
data_saved = np.load("saved_full_solution.npz")
x_pinn_old = data_saved["x"]
t_pinn_old = data_saved["t"]
u_pinn_old = data_saved["u_pinn"]
u_fdm = data_saved["u_fdm"]

# Define spatial grid
xx = np.linspace(0, 1, num=50, dtype=np.float32)

# Time points for visualization
t_test = [200, 500, 2000, 5000]
t_test_indices = [int(t / 10000 * 50) for t in t_test]  # Convert to indices

fig, axes = plt.subplots(1, 4, figsize=(12, 4))  # Single row of 4 plots

for i, t_idx in enumerate(t_test_indices):
    ax = axes[i]
    ax.plot(xx, u_pinn_new[t_idx, :], "r-", label="New PINN")
    ax.plot(xx, u_pinn_old[t_idx, :], "b--", label="Old PINN")
    ax.plot(xx, u_fdm[t_idx, :], "k-", label="FDM")
    ax.set_title(f"t = {t_test[i] * 1e-4:.3f}")
    ax.set_ylim([-0.1, 1.5])

    # Add legend only to the first plot to save space
    if i == 0:
        ax.legend()

plt.tight_layout()
plt.show()
