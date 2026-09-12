import os
import subprocess
import sys


PROJECT_DIR = r"D:\python\HumanChess"

PYTHON_EXE = (
    r"D:\python\HumanChess"
    r"\.venv\Scripts\python.exe"
)


def main():
    os.chdir(PROJECT_DIR)

    subprocess.run(
        [
            PYTHON_EXE,
            "-m",
            "src.uci_engine",
        ]
    )


if __name__ == "__main__":
    main()