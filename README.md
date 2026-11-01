# Mushroom Classification

This project compares two machine-learning models on a mushroom dataset: a
Decision Tree and a Random Forest. Each mushroom is described by 22 categorical
features, such as cap shape, odor, gill color, and habitat. The models predict
whether it is labeled edible or poisonous.

## Getting started

You’ll need Python and scikit-learn. Install scikit-learn if it isn’t already
available in your Python environment:

```bash
python -m pip install scikit-learn
```

Keep `agaricus-lepiota.data` next to the script, then run:

```bash
python train_mushroom.py
```

The script splits the data into training and test sets, fits both models, and
prints their classification reports, confusion matrices, and poisonous-mushroom
sensitivity. It runs locally and does not save model files.

## How the data is handled

The feature values are categories rather than measurements, so the script
one-hot encodes them instead of scaling them. A `?` in the data represents a
missing feature value. Each model pipeline fills those values with the most
frequent category learned from its training data before encoding the features.