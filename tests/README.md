## SPLADE Index tests

This test suite is designed to test the `splade-index` package.

### Core tests
To run the core tests (of library), simply run the following command:

```bash
python -m unittest discover -s tests/core
```

### Numba backend 

To test the numba backend, you have to run:

```bash
python -m unittest discover -s tests/numba
```