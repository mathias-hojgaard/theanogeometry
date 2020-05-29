![](logo/stocso31s.jpg)

# Theano Geometry #

The code in this repository is based on the papers *Differential geometry and stochastic dynamics with deep learning numerics* [arXiv:1712.08364](https://arxiv.org/abs/1712.08364) and *Computational Anatomy in Theano* [arXiv:1706.07690](https://arxiv.org/abs/1706.07690).

### Who do I talk to? ###

Please contact Stefan Sommer *sommer@di.ku.dk*

### Installation Instructions ###

Please use Python 3.X.

#### pip:
Install numpy, scipy, theano, jupyter, matplotlib, multiprocess, sklearn:
```
pip install numpy scipy theano jupyter matplotlib multiprocess sklearn
```
Use e.g. a Python 3 virtualenv:
```
virtualenv -p python3 .
source bin/activate
pip install numpy scipy theano jupyter matplotlib multiprocess sklearn
```
If you don't use a virtual environment, make sure that you are actually using Python 3, e.g. use pip3 instead of pip.

Start jupyter notebook as in
```
export OMP_NUM_THREADS=1; THEANORC=.theanorc jupyter notebook
```
alternatively, to suppress some warnings on e.g. Ubuntu,
```
export THEANO_FLAGS=blas.ldflags="-L/usr/lib/x86_64-linux-gnu -lopenblas"; export OMP_NUM_THREADS=1; THEANORC=.theanorc jupyter notebook
```

Some features, e.g. higher-order landmarks, may require a 'Bleeding-Edge Installation' installation of Theano, see http://deeplearning.net/software/theano/install.html installation instructions.

#### conda:
Install miniconda for Python 3.6 (or higher) from https://conda.io/miniconda.html  
Windows: Open the now installed 'Anaconda Prompt' program.  
Create a new conda environment and activate it by issuing the following commands in the Anaconda prompt:
```
conda create -n theanogeometry python=3
conda activate theanogeometry
```
Use Conda to install the necessary packages:

Linux:
```
conda install git numpy scipy theano jupyter matplotlib multiprocess scikit-learn
```
Windows:
```
conda install git numpy scipy theano m2w64-toolchain mkl-service libpython jupyter matplotlib multiprocess scikit-learn
```
Use git to download Theano Geometry and cd to the directory:
```
git clone https://bitbucket.org/stefansommer/theanogeometry.git
cd theanogeometry
```
Start Jupyter:

Linux (bash):
```
export OMP_NUM_THREADS=1; THEANORC=.theanorc jupyter notebook
```
Windows:
```
set THEANORC=.theanorc 
jupyter notebook
```
Your browser should now open with a list of the Theano Geometry notebooks in the main folder.
