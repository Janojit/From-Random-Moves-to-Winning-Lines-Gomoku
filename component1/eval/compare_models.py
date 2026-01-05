import numpy as np
import matplotlib.pyplot as plt

results = np.load("eval/eval_results.npy", allow_pickle=True).item()

algos = list(results.keys())
wins = [results[a][0] for a in algos]
losses = [results[a][1] for a in algos]
draws = [results[a][2] for a in algos]

# Win Rate Plot
plt.figure()
plt.bar(algos, wins)
plt.title("Win Count Comparison")
plt.savefig("win_count.png")
plt.close()

# Loss Rate Plot
plt.figure()
plt.bar(algos, losses)
plt.title("Loss Count Comparison")
plt.savefig("loss_count.png")
plt.close()

# Draw Rate Plot
plt.figure()
plt.bar(algos, draws)
plt.title("Draw Count Comparison")
plt.savefig("draw_count.png")
plt.close()

print("Comparison plots saved.")
