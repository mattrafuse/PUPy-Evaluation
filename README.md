# PUPy Evaluation

This repository contains the python code used to evaluate the [PUPy](https://github.com/mattrafuse/PUPy) system by Matthew Rafuse, completed for fulfillment of his master's thesis at the University of Waterloo.

## Datasets

We used the [Cambridge/Haggle](https://crawdad.org/cambridge/haggle/20090529/) and [MDC](https://www.idiap.ch/dataset/mdc) datasets. These datasets are not included in this repository for privacy reasons.

## Installation & Run Instruction

If you would like to run this code to double check our results, you'll need to gain access to these datasets, and (optionally, to improve first-run speed) preprocess the data a bit (contact me for details). You then place them into the folders:

- `datasets/mdc` for the MDC dataset (see the source code, make sure the paths match up)
- `datasets/haggle` for the Cambridge/Haggle dataset (again, see the source code)

You will also need to have `matplotlib` and `numpy` installed:

```
pip install numpy matplotlib
```

After that, simply run `python graph.py`, after uncommenting the specific methods you are interested in. The first run will take a *really* long time unless you have a very powerful computer, but the program creates caches for most steps after the first run, making subsequent runs significantly faster. As an example, original parsing of the csv files took 48 hours on a 120 core machine, but after the intermediate files are created, I can reproduce the graphs on my laptop.

## Publications

I'll link my thesis here once it's available! for now, you can contact UWaterloo's [Math Grad Office](mailto:mgo@uwaterloo.ca) for a copy.