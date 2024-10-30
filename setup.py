import setuptools

name = "muTOV"

setuptools.setup(
    name=name,
    version="0.0.1",
    author="Spencer Magnall, Simon Goode, Nikhil Sarin, Paul Lasky",
    author_email="spencer.magnall@monash.edu",
    description="\mu-TOV: Microsecond TOV solver",
    packages=[name],
    package_dir={name: name},
    python_requires='>=3.12',
    install_requires=[
        "bilby",
        "scikit-learn",
        "nestle",
        "tensorflow",
        "lalsuite"]
)
