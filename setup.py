from setuptools import setup, find_packages

package_name = "splade_index"
version = {}
with open(f"{package_name}/version.py", encoding="utf8") as fp:
    exec(fp.read(), version)

with open("README.md", encoding="utf8") as fp:
    long_description = fp.read()

extras_require = {
    "core": ["orjson", "tqdm", "PyStemmer", "numba"],
    "stem": ["PyStemmer"],
    "hf": ["huggingface_hub"],
    "dev": ["black"],
    "selection": ["jax[cpu]"],
    "evaluation": ["pytrec_eval"],
}
# Dynamically create the 'full' extra by combining all other extras
extras_require["full"] = sum(extras_require.values(), [])

setup(
    name=package_name,
    version=version["__version__"],
    author="Yosef Worku Alemneh",
    author_email="",
    url=f"https://github.com/rasyosef/{package_name}",
    description=f"An ultra-fast search index for SPLADE sparse retrieval models.",
    long_description=long_description,
    packages=find_packages(include=[f"{package_name}*"]),
    package_data={},
    install_requires=["scipy", "numpy", "sentence-transformers>=5.0.0"],
    extras_require=extras_require,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    # Cast long description to markdown
    long_description_content_type="text/markdown",
)
