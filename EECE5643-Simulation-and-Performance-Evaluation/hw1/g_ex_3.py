import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dice_results.csv")

# Plot lines for each seed
for seed in df["Seed"].unique():
    subset = df[df["Seed"] == seed]
    plt.plot(subset["NumTrials"], subset["CumulativeEstimate"], marker='o', label=f"Seed={seed}")

plt.xlabel("Cummulative Number of Iterations")
plt.ylabel("Cumulative Estimate of Probability")
plt.legend()
plt.show()
