import time
from scipy.optimize import linprog


def run_simplex(A, b, c):
    tic = time.perf_counter()
    result = linprog(c, A_eq=A, b_eq=b, method="simplex")
    toc = time.perf_counter()

    return result, toc - tic

def test():
    print("Testing linprog on example from class...")

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
    result, time_taken = run_simplex(A_eq, b_eq, c)

    print("Success:", result.success)
    print("Status:", result.message)
    print("Time taken:", time_taken, "seconds")

    if result.success:
        print("Optimal x:", result.x)
        print("Minimum objective value:", result.fun)
        print("Number of pivots:", result.nit)

def experiment():
    ns = [4, 10, 20, 30, 40, 50]
    ms = [2, 6, 10, 14]

    # TODO: run some experiments. Should we randomize the objective/constraints each time or make a list of them
    # for reproducibility?

def main():
    test()
    #  experiment()

if __name__ == "__main__":
    main()