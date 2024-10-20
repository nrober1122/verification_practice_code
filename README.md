# Verification Practice Code 
This repo is meant to be a simple training ground for verification of neural feedback loops (NFLs). Specifically, we will use the auto_LiRPA verification tool to calculate reachable set over-approximations.

# Setup
First, it is recommended to set up a virtual environment. This code was developed with Python 3.10.3, so if that is not the python version natively on your machine, I would recommend using pyenv to set up your virtual environment. A tutorial on how to do that can be found [here](https://realpython.com/intro-to-pyenv/).

Once your virtual environment is active, install the repo's dependencies with the following commands:
```
$ pip install -r actual_requirements.txt
$ pip install -e third_party/auto_LiRPA
```
# Running the Code
If everything is installed correctly, the command 
```
$ python verification_practice/src/test.py --config configs/default_concrete.yaml
```
should give you a `NotImplementedError`. This is becuase there are several things in the file `verification_practice/src/utils/robust_training_utils.py` that need to be implemented. Specifically, there are four problems (lines 112, 135, 275, and 309) each of which need a small amount of code added to complete the file.

Problems 1 and 2 will help train you by directly interfacing with functions from auto_LiRPA (refer to `third_party/auto_LiRPA/examples/simple/mip_lp_solver.py` to see examples on how the relavent functions work), while problems 3 and 4 use the provided code to conduct concrete and symbolic reachability analyses, respectively.
Solutions to each problem can be found in `solutions.txt` if you are really stuck. Reach out to me if you have any questions!

Once you are done completing the code, the command 
```
$ python verification_practice/src/test.py --config configs/default_concrete.yaml
```
should produce an image that looks like this:

<img src="images/tutorial_1.png" width="600">

Similarly, the command 
```
$ python verification_practice/src/test.py --config configs/default_symbolic.yaml
```
should produce an image that looks like this:

<img src="images/tutorial_2.png" width="600">

Good luck!
