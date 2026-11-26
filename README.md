# CAFE (Component Automated Feature Engineer)

CAFE automates data preprocessing and feature engineering for machine learning on tabular data. You hand it a pandas DataFrame and a target column, and it returns the transformed DataFrame. It also checks, with cross-validation, whether the transformation made a model better or worse than the raw data.

## Install

Needs Python 3.7 or newer.

```bash
pip install cafe-autofe
```

## Use

Fit a pipeline with the default settings:

```python
import pandas as pd
from cafe import DataPipeline

df = pd.read_csv("dataset.csv")

pipeline = DataPipeline()
df_transformed = pipeline.fit_transform(df, target_col="target")

pipeline.save("pipeline_model")
```

Let the `Explorer` search for the best configuration first:

```python
import pandas as pd
from cafe import DataPipeline, Explorer

df = pd.read_csv("dataset.csv")

explorer = Explorer(target_col="target")
explorer.analyze_transformations(df)
best_config = explorer.get_best_pipeline_config()

pipeline = DataPipeline(
    preprocessor_config=best_config.get("preprocessor_config", {}),
    feature_engineer_config=best_config.get("feature_engineer_config", {}),
)
df_transformed = pipeline.fit_transform(df, target_col="target")
```

## Components

| Class | What it does |
|---|---|
| `PreProcessor` | Cleans the data: missing values, outliers, encoding and scaling. |
| `FeatureEngineer` | Creates and selects features. |
| `PerformanceValidator` | Compares a model on the raw and on the transformed data with cross-validation. |
| `DataPipeline` | Runs the three above in sequence, with `fit`, `transform`, `save` and `load`. |
| `Explorer` | Tries transformation combinations and returns the best pipeline configuration. |
| `ReportDataPipeline` | Reports missing values, outliers, feature importance and the transformations applied. |
| `ReportVisualizer` | Plots those reports, plus distributions and the correlation matrix. |

Every configuration option, with complete examples for missing values, high dimensionality, high-cardinality categories and time data, is in the [Portuguese reference](docs/README.pt-BR.md). There is also a [tutorial](tutorial.md), in Portuguese.

## Status

The code has not changed since March 2025.

## License

MIT. See [LICENSE](LICENSE).
