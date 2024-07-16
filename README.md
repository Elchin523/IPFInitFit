
# IPFInitFit

This package provides functionalities for Iterative Proportional Fitting (IPF) with optimal initial weights.

## Installation

You can install the package using:

```bash
pip install git+https://github.com/yourusername/IPFInitFit.git
```

## Usage

```python
import pandas as pd
from IPFInitFit import IPF

# Initialize the IPF class
ipf = IPF()

# Your data and constraints
data = pd.DataFrame(...)  # Your input data
constraints = [...]  # Your constraints

# Apply weighting
data_weighted = ipf.apply_weighting(data, constraints)

# Check results
ipf.check_results(data, data_weighted, constraints)
```
# IPFinFit
