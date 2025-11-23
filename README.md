# FeatBoost
Python implementation of FeatBoost. See the [paper](https://doi.org/10.1016/j.eswa.2021.115895) for details.

## Demo
This demo compares featboost to feature ranking of XGBoost on the Madelon benchmark dataset.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eEySuIAJzmlNOChfLwEqJFKKbGNVYMwJ)

## Install

```shell
git submodule add https://github.com/EdwardChanCH/FeatBoost-Fork.git
git submodule init
git submodule update
```

## Example Code
```python
# Load Python file
import sys
feat_boost_relpath = "./FeatBoost-Fork/featboost" # Relative path from the notebook files to the 'featboost' folder.
if feat_boost_relpath not in sys.path:
    sys.path.append(feat_boost_relpath)

# Import FeatBoost
from feat_boost import FeatBoostClassifier

# Set up input data
X_fs = X_train # Training features (dataset)
y_fs = y_train # Training targets (dataset)

# Set up estimator
fs_estimator = XGBRegressor(
    max_depth=3,
    learning_rate=0.1,
    n_estimators=20,
    objective="reg:squarederror",
    n_jobs=1,
    random_state=42,
)

# Set up feature selection method
# Note: "siso_ranking_size" (output size) must be <= #features in the input dataframe (input size).
#       "siso_order" must be < "siso_ranking_size".

print("--- FeatBoost Started ---")

# Note: FeatBoost is originally intended for XGBoost Classifier, but now converted to accept XGBoost Regressor.
fs = FeatBoostClassifier(
    estimator=fs_estimator,
    siso_ranking_size=5,
    loss="adaboost",
    metric="acc",
    verbose=2
)

# Run feature selection
fs.fit(X_fs, y_fs)

print("--- FeatBoost Ended ---")
print(fs.feature_importances_array_)
```
