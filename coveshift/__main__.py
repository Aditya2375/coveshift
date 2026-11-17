import argparse
import json
from .core import analyze, settings


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic factor covariance and Euler attribution"
    )
    parser.add_argument("--scenario", type=int, default=0)
    parser.add_argument("--shrink", type=int, default=0)
    parser.add_argument("--residual", choices=("diagonal", "full"), default="diagonal")
    parser.add_argument("--weights", nargs=5, type=int, default=settings()["weights"])
    args = parser.parse_args()
    try:
        result = analyze(vars(args))
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
