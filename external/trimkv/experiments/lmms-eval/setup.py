from setuptools import setup, find_packages

setup(
    name="lmms_eval",            # arbitrary distribution name
    version="0.1.0",
    packages=find_packages(
        where=".",
        include=["lmms_eval*", "pruning_baseline_src*"],  # include both
        exclude=["tests*", "docs*"]
    ),
    license_files=["LICENSE"],
)