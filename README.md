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

There is an issue with the dropdown in the side bar for most recent versions
of `sphinx` and `sphinx-gallery`. Thus, these have been fixed in the requirements
to the following versions: `sphinx==5.3.0` and `sphinx-gallery==0.10.0`.

### Adding a new project

Create a new folder `foldername` and a `foldername/scriptname.py` script as shown below

```python
"""
Title of the script
========================

Description of the script

"""
# Code
print("Works!")
```


### Creating docs

To include all scripts within `foldername` in the documentation, the 
folder must contain a `README.rst` file as shown below.
 
 ```sh
"""
Sidebar title
=============

```
 
Also, remember that the script name must start with `plot_<scriptname>.py` 
so that the graphical output is also included in the documentation. See 
`sphinx` and `sphinx-gallery`  for more information.


Run the following command whithin `main/docs`

```sh
$ make github    # Generate sphinx docs.
```

And commit all the changes in gh-pages.
