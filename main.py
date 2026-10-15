import time
import json
import numpy as np
from pathlib import Path
from scipy.optimize import linprog


def run_simplex(A, b, c, bounds=(0, None)):
    tic = time.perf_counter()
    result = linprog(c, A_eq=A, b_eq=b, method="simplex", bounds=bounds)
    toc = time.perf_counter()

    data = dict(result)
    data["time_taken"] = toc - tic
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            data[k] = v.tolist()
    # data["A"] = A.tolist() if isinstance(A, np.ndarray) else A
    # data["b"] = b.tolist() if isinstance(b, np.ndarray) else b
    # data["c"] = c.tolist() if isinstance(c, np.ndarray) else c

    return data

def test_from_class():
    print("Running Simplex on example from class...")

    # Minimize 7x_1 + 4x_2 + 6x_3 + x_4
    c = [7, 4, 6, 1]

    # Constraints:
    #   x_1 + 2x_2 -  x_3 -  x_4 = 1
    #  -x_1 - 5x_2 + 2x_3 + 3x_4 = 1
    #  x_i \geq 0
    A_eq = [
        [1, 2, -1, -1],
        [-1, -5, 2, 3]
    ]
    b_eq = [1, 1]

    # Nonnegative variables: x >= 0, y >= 0
    bounds = (0, None)

    # Solve using the simplex tableau method
    result = run_simplex(A_eq, b_eq, c, bounds=bounds)

    print("Success:", result["success"])
    print("Status:", result["message"])
    print("Time taken:", result["time_taken"], "seconds")

    if result["success"]:
        print("Optimal x:", result["x"])
        print("Minimum objective value:", result["fun"])
        print("Number of pivots:", result["nit"])

    print()

def random_lp(m, n, seed=2016, coeff_bounds=(-10, 11)):
    rng = np.random.default_rng(seed)

    A = rng.integers(*coeff_bounds, size=(m, n))
    b = rng.integers(*coeff_bounds, size=m)
    c = rng.integers(*coeff_bounds, size=n)

    return A, b, c

def experiment(seed=2016):
    tic = time.time()

    ns = [4, 10, 20, 30, 40, 50]
    ms = [2, 6, 10, 14]
    bounds = (0, None)

    results = []

    for n in ns:
        for m in ms:
            print(f"Running experiment with n={n}, m={m}, seed={seed}...")

            A, b, c = random_lp(m, n, seed=seed)
            result = run_simplex(A, b, c, bounds=bounds)
            results.append(result)

    data_file = Path(f"results/results_{seed}.json")
    data_file.parent.mkdir(parents=True, exist_ok=True)

    with open(data_file, "w") as f:
        json.dump(results, f, indent=2)

    toc = time.time()
    print(f"Experiment completed in {toc - tic:.2f} seconds. Results saved to {data_file}.")
    print(f"Number of Results: {len(results)}")
    print()

def main():
    seeds = range(2000, 2101, 4)
    for seed in seeds:
        experiment(seed=seed)

    test_from_class()

if __name__ == "__main__":
    main()