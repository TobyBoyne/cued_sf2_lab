# Part IIA Project: SF2: Image Processing

[![codecov](https://codecov.io/gh/sigproc/cued_sf2_lab/branch/master/graph/badge.svg)](https://codecov.io/gh/sigproc/cued_sf2_lab)
[![Gitpod Ready-to-Code](https://img.shields.io/badge/Gitpod-ready--to--code-blue?logo=gitpod)](https://gitpod.io/from-referrer)

This repository contains the Python package and Jupyter Notebooks for the SF2 lab project in the Cambridge University Engineering department.

***Note that the new Python version of this lab is still in development. The notebooks are not a replacement for the PDF handouts.***

~~***It is likely that the notebooks will be updated midway through the course.***~~
The Python code in `cued_sf2_lab` has been updated, and a new notebook for section 12 has been created.

To get started, you should:

* **If using the DPO computers**:
  * It is **strongly** advisible to boot into Linux, as `git` will already be installed there.
  * You should use the "Anaconda terminal" not the usual terminal, as this will have a more recent version of Python.
  * If you are using Windows in the DPO:
    * You might have file path issues unless you work on the `U:` or `Z:` drives. To use `cd` in a command prompt to change drives, first type `U:` and then `cd U:/path/to/sf2`.
    * You might find you are unable to install things with `pip`, and that your plots don't show up correctly, due to an old version of Anaconda being used that you will not be able to update.
* Have a recent version of python + Jupyter installed.
  Check that `python --version` emits what you expect it to.
* `git clone` this repository (recommended). If you do not have git installed, you can download and extract the zip from the top of the github page; but this will make it harder for you to get updated versions.
* Open a command prompt in the folder you downloaded the code to, and run `python -m pip install -e . --user --upgrade`.
  This will install various dependencies, and a `cued_sf2_lab` python package containing a collection of helper functions.
  * **If this fails, make sure to speak to a demonstrator** rather than just continuing anyway; it likely means there is a bigger problem with your python system
* Open the notebooks (`ipynb` files) in the root of this repository.
  * **Do not "upload" the notebooks into Jupyter**, this will just make a copy and leave you with two different copies on your system!

## FAQ

1. **Why aren't my matplotlib color bar plots showing up?**  
   Likely you are not using the latest matplotlib.
   You can find out the version with `import matplotlib; matplotlib.__version__` inside Jupyter.
   To update, try `pip install --upgrade --user matplotlib`.
   
2. **Why aren't the image plots showing up in VSCode**?
   The notebooks currently contain `%matplotlib nbagg`, which works in Jupyter but not VSCode.
   If you replace it with `%matplotlib widget`, then you will get interactive plots that work in VSCode.
   Note that you will need to:
    * Have the latest (possibly even the preview) `jupyter-vscode` extension
    * Answer "yes" to the popup that appear in the corner of your screen about downloading additional files.
    * Restart VSCode multiple times after updating these components.

3. **Why isn't `%matplotlib widget` working**?
   This requires very recent versions of the `notebook` and `matplotlib` packages.
   If you get an error about `ipympl` not existing, then you did not follow one of the steps above.
   Likely, you have found yourself on a system that refuses to be updated (like the DPO computers on windows), and there is nothing you can do.

## Note for demonstrators

This software consists of two repos, which share this README.

* https://github.com/sigproc/cued_sf2_lab
* https://github.com/sigproc/cued_sf2_lab-answers

If you are a student, you will only have access to the former!
If you are a demonstrator, you should request access to the latter.
The answers repository generates the other repository automatically.

More information for demonstrators can be found in [the demonstrator readme](https://github.com/sigproc/cued_sf2_lab-answers/blob/main/README-demonstrators.md).
