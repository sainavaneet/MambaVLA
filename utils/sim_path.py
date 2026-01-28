import os

FRAMEWORK_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir)
)


def sim_framework_path(*args) -> str:
    abs_path = os.path.abspath(os.path.join(FRAMEWORK_DIR, *args))
    return abs_path
