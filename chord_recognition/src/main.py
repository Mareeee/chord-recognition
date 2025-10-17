import warnings

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning
)

from actions import menu

if __name__ == "__main__":
    menu()
