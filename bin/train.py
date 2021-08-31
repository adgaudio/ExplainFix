"""
Main entrypoint to train a fixed filters model.
"""
from dw2.configs.fixed_filters import main, arg_parser


if __name__ == "__main__":
    main(arg_parser().parse_args().__dict__)
