import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "NumpifyML"
AUTHOR_USER_NAME = "AbhayaHanuma"
SRC_REPO = "numpifyml"
AUTHOR_EMAIL = "abhayabhi987@gmail.com"
DIR = "numpifyml"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for Machine Learning",
    long_description=long_description,
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": f"{DIR}"},
    packages=setuptools.find_packages(where=f"{DIR}")
)