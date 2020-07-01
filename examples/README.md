# highdicom examples

A set of [Jupyter notebook](https://jupyter-notebook.readthedocs.io/en/stable/) examples that demo various aspects of the library.

## Usage

Build and run container image using [Docker](https://www.docker.com/):

```none
docker build . -t highdicom/examples:latest
docker run --rm --name highdicom_examples -p 8888:8888 highdicom/examples:latest
```

After running the above commands, following the instructions printed into the standard output stream to access the notebooks in your browser.
