"""Run the three paper examples.

python src/run_all_examples.py [--example 1|2|3]
Outputs go to results_paper/example_0N/ at the package root.
"""

import sys
import os
import argparse

_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.join(_SRC, "..", "..", "pykan"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--example', type=int, choices=[1, 2, 3])
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    results = {}

    if args.example is None or args.example == 1:
        print("\n=== Example 01: forward relaxation (PIKAN) ===")
        from example_01_viscoelastic_forward_relaxation import main as main_01
        results['example_01'] = main_01()
        plt.close('all')

    if args.example is None or args.example == 2:
        print("\n=== Example 02: forward general loading (L1 vs PIKAN) ===")
        from example_02_viscoelastic_forward_general_loading import main as main_02
        results['example_02'] = main_02()
        plt.close('all')

    if args.example is None or args.example == 3:
        print("\n=== Example 03: parameter inversion (Standard KAN vs PEKAN) ===")
        from example_03_viscoelastic import main as main_03
        results['example_03'] = main_03()
        plt.close('all')

    print("\nDone. Outputs under results_paper/example_0N/{figures,data}.")
    return results


if __name__ == "__main__":
    main()
