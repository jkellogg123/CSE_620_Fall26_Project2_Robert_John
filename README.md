# CSE 620 Simplex Project

## Setup

This project uses SciPy's implementation of the simple tableau method. Newer versions
of SciPy opt for a more efficient and complex version of the algorithm, so we'll need to revert
to a version before 1.9.0. The latest such version is 1.8.1, which only supports up to Python 3.10,
so we'll need an environment for that.

**Conda:**

```bash
conda create -n cse620 python=3.10
conda activate cse620
pip install -r requirements.txt