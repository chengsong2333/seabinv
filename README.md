# seabinv v1.0

seabinv, based on BayHunter [http://doi.org/10.5880/GFZ.2.4.2019.001](http://doi.org/10.5880/GFZ.2.4.2019.001), is an open source Python tool to perform a parallel tempering McMC trans-dimensional Bayesian inversion of sea level. The algorithm follows a data-driven strategy and solves for the age, the number of segment, noise.

### Features
* Parallel tempering
* Trans-dimensional Bayesion inversion

### Citation

coming soon

### Application examples


### Comments and Feedback

Looking forward to your feedback


### Who am I?

I am Cheng Song. I am now a PhD student of Seoul National University. [Contact me](songcheng@snu.ac.kr).

## Quick start

### Requirements
* matplotlib
* numpy
* PyPDF2
* configobj
* zmq
* Cython

### Installation (compatible with Python 2 and 3)*

*Although BayHunter is currently compatible with Python 2 and 3, we recommend you to upgrade to Python 3, as the official support for Python 2 has stopped in January 2020.

```sh
git clone https://github.com/chengsong2333/seabinv
cd seabinv
python setup.py install
```

### Documentation and Tutorial

The documentation to seabinv offers background information on the inversion algorithm, the parameters and usage of seabinv and seabinv (tutorial). Refer to the [documentation here](https://jenndrei.github.io/BayHunter/) or download the [PDF](https://github.com/jenndrei/BayHunter/blob/master/documentation/BayHunter_v2.1_documentation.pdf).

An example inversion can be found in the **tutorial folder**.
The file to be run, `tutorialhunt.py`, is spiked with comments.

### Resources

* Algorithm: based on the work of [Bodin et al., 2012](https://doi.org/10.1029/2011JB008560) and [http://doi.org/10.5880/GFZ.2.4.2019.001](http://doi.org/10.5880/GFZ.2.4.2019.001).
# seabinv
