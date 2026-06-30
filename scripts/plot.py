import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

csv_path = Path(sys.argv[1]).expanduser().resolve()
df = pd.read_csv(csv_path).sort_values("gen")

fig = plt.figure(figsize=(9, 6.5))

ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(df["gen"], df["best"], label="Best", linewidth=2)
ax1.plot(df["gen"], df["mean"], label="Mean", linestyle="--")
ax1.set_xlabel("Generation"); 
ax1.set_ylabel("Fitness")
ax1.set_title("Fitness Over Generations");
ax1.legend();
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(df["gen"], df["diversity"], label="Diversity")
ax2.set_xlabel("Generation");
ax2.set_ylabel("Diversity")
ax2.set_title("Population Diversity Over Time");
ax2.legend();
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(csv_path.with_name(f"{csv_path.stem}_summary.png"), dpi=200)

