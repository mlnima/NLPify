import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


with open("requirements.txt", "r") as requirements_file:
    packages = requirements_file.readlines()

for package in packages:
    install(package.strip())
