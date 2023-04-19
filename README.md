## About The Project

This repository contains a diverse collection of sample scripts that have been 
gathered from a variety of sources, including official documentation sites or 
common sites where programmers post question, answers and/or tutorials. Additionally, 
some of the scripts have been edited by the repository owner to make them more 
comprehensive or tailored to specific use cases.

## Getting Started

### Prerequisites

See the `requirements.txt` file.

* `matplotlib`: to create the matplotlib figures.
* `plotly`: to create dynamic figures.
* `tableone`: to create the demographics tables.
* `scikits`: to create ML models.
* ...

Install libraries as follows

```sh
$ python -m pip install -r requirements.txt
```

### Adding a new project

Create a new folder `foldername`. 

Create a `foldername/scriptname.py` script which starts with the following...

```python
"""
Title of the script
========================

Description of the script

"""
# Code
print("Works!")
```

To include scripts within the folder when creating the docs include 
a `README.rst` inside `foldername`. Remember that the script name must
start with `plot_` to include the graphical outputs. See `sphinx` and
`sphinx-gallery` for more information.


### Creating docs

Run the following command whithin `main/docs`

```sh
$ make github    # Generate sphinx docs.
```

And commit all the changes in gh-pages.
