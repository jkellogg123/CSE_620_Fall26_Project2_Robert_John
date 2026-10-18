import time
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import linprog


def run_simplex(A, b, c, bounds=(0, None)):
    tic = time.perf_counter()
    result = linprog(c, A_eq=A, b_eq=b, method="simplex", bounds=bounds)
    toc = time.perf_counter()

    data = dict(result)
    data["time_taken"] = toc - tic
    data["A"] = A
    data["b"] = b
    data["c"] = c

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

    data_file = Path(f"results/seed_{seed}/results_data.json")
    data_file.parent.mkdir(parents=True, exist_ok=True)

    with open(data_file, "w") as f:
        json.dump(results, f, indent=2)

    toc = time.time()
    print(f"Experiment completed in {toc - tic:.2f} seconds. Results saved to {data_file}.")
    print(f"Number of Results: {len(results)}")
    print()

def tabulate_latex():
    results_dir = Path("results")
    
    if not results_dir.exists():
        print("Results directory does not exist. Please run experiments first.")
        return

    success_results = []
    success_dicts = []

    for seed_dir in sorted(results_dir.iterdir()):
        if not seed_dir.name.startswith("seed_"):
            continue

        data_file = seed_dir / "results_data.json"
        with open(data_file, "r") as f:
            results = json.load(f)

        # TODO: Make Latex table from this
        rows = []
        for lp in results:
            m, n = np.array(lp["A"]).shape

            success = lp["success"]
            if not success:
                message = lp["message"]
                if "infeasible" in message:
                    success = "Infeasible"
                elif "unbounded" in message:
                    success = "Unbounded"
            if success is True:
                success_dicts.append(lp)

            time_taken = lp["time_taken"]
            pivots = lp["nit"] if success is True else "N/A"
            cost = lp["fun"] if success is True else "N/A"
            if isinstance(cost, (int, float)):
                cost = f"{cost:.2f}"

            rows.append(
                f"{n} & {m} & {success} & {time_taken:.4f} & {pivots} & {cost} \\\\"
            )

            if success is True:
                success_results.append(
                    f"{n} & {m} & {success} & {time_taken:.4f} & {pivots} & {cost} \\\\"
                )

        latex_table = r"""\begin{table}[ht]
    \centering
    \begin{tabular}{rrrrrr}
        \toprule
        Num Variables & Num Constraints & Success & Time & Pivots & Cost \\
        \midrule
        """ + "\n\t\t".join(rows) + r"""
        \bottomrule
    \end{tabular}
    \caption{Linear programming results}
    \label{tab:lp-results}
\end{table}"""

        table_file = seed_dir / "results_table.tex"
        with open(table_file, "w") as f:
            f.write(latex_table)
        # print(latex_table)
        print(f"Saved LaTeX table to \"{table_file}\"")

    success_table = r"""\begin{table}[ht]
    \centering
    \begin{tabular}{rrrrrr}
        \toprule
        Num Variables & Num Constraints & Success & Time & Pivots & Cost \\
        \midrule
        """ + "\n\t\t".join(success_results) + r"""
        \bottomrule
    \end{tabular}
    \caption{Linear programming results}
    \label{tab:lp-results}
\end{table}"""

    table_file = results_dir / "successes" / "results_table.tex"
    table_file.parent.mkdir(parents=True, exist_ok=True)
    with open(table_file, "w") as f:
        f.write(success_table)
    # print(success_table)
    print(f"Saved LaTeX table to \"{table_file}\"")

    data_file = results_dir / "successes" / "results_data.json"
    data_file.parent.mkdir(parents=True, exist_ok=True)

    with open(data_file, "w") as f:
        json.dump(success_dicts, f, indent=2)

def plot_results():
    success_dir = Path("results") / "successes"
    data_file = success_dir / "results_data.json"
    if not data_file.exists():
        print("No success data found. Please run experiments first.")
        return

    with open(data_file, "r") as f:
        results = json.load(f)

    n_vars = [np.array(lp["A"]).shape[1] for lp in results]
    n_constraints = [np.array(lp["A"]).shape[0] for lp in results]
    times = [lp["time_taken"] for lp in results]
    problem_size = [n * m for n, m in zip(n_vars, n_constraints)]
    costs = [lp["fun"] for lp in results]
    pivots = [lp["nit"] for lp in results]

    plots = [
        (n_vars, n_constraints, "Number of Variables", "Number of Constraints",
        "Problem Dimensions", "variables_vs_constraints.png"),
        (n_vars, times, "Number of Variables", "Time (seconds)",
        "Solve Time vs Variables", "time_vs_variables.png"),
        (n_constraints, times, "Number of Constraints", "Time (seconds)",
        "Solve Time vs Constraints", "time_vs_constraints.png"),
        (problem_size, times, "Variables × Constraints (n × m)", "Time (seconds)",
        "Solve Time vs Problem Size", "time_vs_nm.png"),
        (n_vars, costs, "Number of Variables", "Solution Cost", 
        "Solution Cost vs. Variables", "cost_vs_variables.png"),
        (n_constraints, costs, "Number of Constraints", "Solution Cost", 
        "Solution Cost vs. Constraints", "cost_vs_constraints.png"),
        (problem_size, costs, "Variables × Constraints (n × m)", "Solution Cost", 
        "Solution Cost vs. Problem Size", "cost_vs_nm.png"),
        (pivots, times,
        "Number of Pivots", "Time (seconds)", "Solve Time vs. Number of Pivots", "time_vs_pivots.png"),
    ]

    for x, y, xlabel, ylabel, title, filename in plots:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x, y, alpha=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(success_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

def table_plots():
    tabulate_latex()
    plot_results()


def main():
    # seeds = range(2000, 2101, 4)
    # for seed in seeds:
    #     experiment(seed=seed)

    # test_from_class()

    table_plots()

if __name__ == "__main__":
    main()