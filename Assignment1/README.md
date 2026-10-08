# CS5590 Foundations of Machine Learning: Assignment 1

**Submitted By:**
- Manoj Kumar V K (CS26RESCH11009)
- Suman Das (CS26RESCH02004)

Q1 to Q3 are coded in the notebook. Q4 and Q5 are pen-and-paper derivations, so they're in the PDFs rather than the code.

## Layout

```text
Assignment1/
├── README.md
├── A1_CS26RESCH11009_CS26RESCH02004.ipynb
├── A1_CS26RESCH11009_CS26RESCH02004.pdf
└── data/
    ├── mess_data.csv
    ├── mnist/
    │   ├── train-images.idx3-ubyte
    │   ├── train-labels.idx1-ubyte
    │   ├── t10k-images.idx3-ubyte
    │   └── t10k-labels.idx1-ubyte
    ├── wine/
    │   ├── winequality-red.csv
    │   └── winequality-white.csv
    └── (handwritten solution PDFs)
```

The notebook reads everything through relative paths, so keep `data/` next to it and don't rename anything.

## Getting the data

`mess_data.csv` is already included. It has 14 days of meal records that the two of us logged separately: date, day, meal, time of the meal, who logged it, day type, crowd level, waiting time, and time taken to finish.

MNIST and Wine Quality are not included, so you need to download them:

- MNIST: https://www.kaggle.com/datasets/hojjatk/mnist-dataset/data. Put the four IDX files in `data/mnist/`. Leave them as IDX, the notebook parses them directly and doesn't want CSVs.
- Wine Quality: https://archive.ics.uci.edu/dataset/186/wine+quality. Put both the red and white CSVs in `data/wine/`. Q3 needs both.

## Setup

I used Python 3.11.0. No GPU needed.

```bash
conda create -n foml python=3.11.0 -y
conda activate foml
pip install numpy pandas matplotlib scipy scikit-learn mord jupyter
jupyter notebook
```

If you don't use conda, `python -m venv foml` and then activating it (`source foml/bin/activate` on Linux/macOS, `foml\Scripts\activate` on Windows) works the same way. The pip line is identical.

Open the notebook and run it top to bottom. The main random seed is `11009`.

## What's in each question

**Q1. mess completion time:** Some EDA on how long a meal takes versus meal type, day, person, crowd level and so on. I then fit a Gamma regression with a log link by maximum likelihood, since the target (`time_to_finish`) is a positive duration. 80/20 split, and I compare it against plain linear regression on MAE and RMSE.

**Q2. PCA on MNIST:** Pixels scaled to [0, 1], PCA down to two components, and the variance explained by each. The test set is projected into the same 2D space and plotted by digit. I also reconstruct a few images from just two components to show how much gets lost, and discuss the class overlap.

**Q3. Ordinal regression on wine quality:** Red and white are merged, with a binary `wine_type` column added. Features are standardised, 80/20 split, and I use `LogisticAT` from `mord`, choosing the regularisation strength by 5-fold CV. Again compared with linear regression. The theory part of Q3 is in the handwritten PDFs.

**Q4. Heteroscedastic least squares:** Derivation only: likelihood and prior, the ML and MAP objectives, how ML turns into weighted least squares, and the closed-form minimiser.

**Q5. logistic regression:** Derivation only: gradient and Hessian of the error function, the Newton-Raphson update, why that is the same as IRLS (weighted least squares each step), and why the error function is convex.


## References

- MNIST: https://www.kaggle.com/datasets/hojjatk/mnist-dataset/data
- Wine Quality: https://archive.ics.uci.edu/dataset/186/wine+quality
- scikit-learn PCA docs: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- McCullagh, P. *Regression Models for Ordinal Data*: https://www.jstor.org/stable/2984952