import numpy as np

results = np.load("eval/tournament_results.npy", allow_pickle=True).item()

print("\n===== FINAL SCOREBOARD =====")
print(f"{'Agent':12s} | Wins | Losses | Draws")
print("-" * 36)

for agent, r in results.items():
    print(f"{agent:12s} | {r['win']:4d} | {r['loss']:6d} | {r['draw']:5d}")
