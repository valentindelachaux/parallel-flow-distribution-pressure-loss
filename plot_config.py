import repo_config as rconfig

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
import matplotlib_inline

import scienceplots

matplotlib_inline.backend_inline.set_matplotlib_formats('retina')
plt.style.use(['science'])

def to_percent(y, _):
    return f"{y * 100:.0f}\%"

# colors = ['olive', 'green', 'blue', 'orange', 'red', 'grey','yellow','bordeaux']
colors = ['olive','orange','blue','red','green','grey']
colors70 = [rconfig.get_hex_code(color, 70) for color in colors]
colors = [rconfig.get_hex_code(color, 100) for color in colors]
linestyles = ['-', '--',  (5, (10, 3)), '-.', (0, (1, 1)), (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5)), (0, (3, 5, 1, 5))]
markers = ['o', 's', 'd', 'X', 'v', '^', 'P', 'H']