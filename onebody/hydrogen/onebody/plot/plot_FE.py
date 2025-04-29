import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import plotsetting as ps

# Load data (example data)
data = np.loadtxt("site_fsE.txt")
site = data[:, 0]
dmrg_E = data[:, 1]    # average 100 times run 
exact_E = data[:, 2]

# Create figure and axis with adjusted size

fig, ax = plt.subplots()

# Scatter plot for Intel Fortran and one-site TDVP
ax.scatter(site, dmrg_E, marker='*', label="DMRG", color='red', alpha = 1, s=100)
ax.scatter(site, exact_E, marker='x', label=r"Exact Func", color='black', s=100)
#ax.scatter(site, gd, marker='o',  label=r"GD, $D_{\psi}$ = 20",  color='red', s=100)s

ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# Set labels and title
ax.set_xlabel(r'Number of qubits', fontsize=16)
ax.set_ylabel(r'Ground Energy', fontsize=16)
#ax.set_yscale('log')
ax.legend(loc="best", fontsize=16)
ax.text(0.1, 0.9, "(a)", fontsize=25, transform=ax.transAxes)
ps.set([ax])
fig.savefig("2D_log_inset.pdf", transparent=False)
#figins.savefig("2D_linear_inset.pdf", transparent=False)
plt.show()



