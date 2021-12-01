from setuptools import setup

VERSION = "0.0.1"

long_description = ""
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

license = ""
with open("LICENSE.md", "r", encoding="utf-8") as fh:
    license = fh.read()

requirements = []
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.readlines()

extras = {
    "sb3": ["stable-baselines3"],
    "rllib": ["ray[rllib]==0.8.5", "tensorflow>=2.6.0"],
    "dev": ["pytest", "black", "isort"],
}

extras["all"] = extras["sb3"] + extras["rllib"]


setup(
    name="social-dilemmas",
    version=VERSION,
    description="Sequential Social Dilemma Environments",
    url="https://github.com/eugenevinitsky/sequential_social_dilemma_games",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license=license,
    packages=["social_dilemmas"],
    install_requires=requirements,
    extras_require=extras,
    python_requires=">=3.7, <3.10",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
