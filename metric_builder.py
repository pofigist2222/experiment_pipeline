import yaml
import config
import abc
import pandas as pd
import numpy as np
from yaml.loader import SafeLoader
from os import listdir


def _load_yaml_preset(preset="todo"):
    preset_path = config.PATH_METRIC_CONFIGS + preset
    metrics_to_load = listdir(preset_path)
    metrics = []
    for metric in metrics_to_load:
        with open(preset_path + "/" + metric) as f:
            metrics.append(yaml.load(f, Loader=SafeLoader))
    return metrics


class Metric:
    def __init__(self, metric_config):
        self.name = metric_config.get("name", config.DEFAULT_VALUE)
        self.type = metric_config.get("type", config.DEFAULT_METRIC_TYPE)
        self.level = metric_config.get("level", config.DEFAULT_UNIT_LEVEL)
        self.estimator = metric_config.get("estimator", config.DEFAULT_ESTIMATOR)
        self.numerator = metric_config.get("numerator", config.DEFAULT_VALUE)
        self.numerator_aggregation_field = self.numerator.get("aggregation_field", config.DEFAULT_VALUE)
        self.denominator = metric_config.get("denominator", config.DEFAULT_VALUE)
        self.denominator_aggregation_field = self.denominator.get("aggregation_field", config.DEFAULT_VALUE)
        numerator_aggregation_function = self.numerator.get("aggregation_function", config.DEFAULT_VALUE)
        denominator_aggregation_function = self.denominator.get("aggregation_function", config.DEFAULT_VALUE)
        self.numerator_aggregation_function = self._map_aggregation_function(numerator_aggregation_function)
        self.denominator_aggregation_function = self._map_aggregation_function(denominator_aggregation_function)

    @staticmethod
    def _map_aggregation_function(aggregation_function):
        mappings = {
            "count_distinct": pd.Series.nunique,
            "sum": np.sum
        }
        if aggregation_function == config.DEFAULT_VALUE:
            raise ValueError("No aggregation_function found")

        agg_func = mappings[aggregation_function]
        if aggregation_function not in mappings.keys():
            raise ValueError(f"{aggregation_function} not found in mappings")
        return agg_func


class CalculateMetric:
    def __init__(self, metric: Metric):
        self.metric = metric

    def __call__(self, df):
        return df.groupby([config.VARIANT_COL, self.metric.level]).apply(
            lambda df: pd.Series({
                "num": self.metric.numerator_aggregation_function(df[self.metric.numerator_aggregation_field]),
                "den": self.metric.denominator_aggregation_function(df[self.metric.denominator_aggregation_field]),
                "n": pd.Series.nunique(df[self.metric.level])
            })
        ).reset_index()


