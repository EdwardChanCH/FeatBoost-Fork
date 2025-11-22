# FeatBoost
Python implementation of FeatBoost. See the [paper](https://doi.org/10.1016/j.eswa.2021.115895) for details.

## Demo
This demo compares featboost to feature ranking of XGBoost on the Madelon benchmark dataset.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eEySuIAJzmlNOChfLwEqJFKKbGNVYMwJ)

## Usage
```shell
# New version
pip install git+https://github.com/EdwardChanCH/FeatBoost-Fork.git
# Old version
pip install git+https://github.com/amjams/FeatBoost.git
```

Or just clone the repo (recommended for now)

```shell
# New version
git clone https://github.com/EdwardChanCH/FeatBoost-Fork.git
# Old version
git clone https://github.com/amjams/FeatBoost.git
```

Basic usage of the model (see the demo above for more details).
```python
import sys
feat_boost_relpath = "./FeatBoost-Fork/featboost" # Relative path from the notebook files.
if feat_boost_relpath not in sys.path:
    sys.path.append(feat_boost_relpath)
from feat_boost import FeatBoostClassifier

clf = FeatBoostClassifier()
clf.fit(X, y)
clf.feature_importances_
```
