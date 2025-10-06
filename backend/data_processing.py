"""Data ingestion and preprocessing utilities for the Irish house price gap pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests
from pyjstat import pyjstat

CSO_DATA_URL = (
    "https://ws.cso.ie/public/api.restful/"
    "PxStat.Data.Cube_API.ReadDataset/HPA02/JSON-stat/2.0/en"
)


@dataclass
class PriceGapDataset:
    """Container for the processed dataset."""

    prices: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series


def fetch_cso_dataset(url: str = CSO_DATA_URL, timeout: int = 30) -> pd.DataFrame:
    """Fetch the raw dataset from the CSO API and return it as a DataFrame."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    dataset = pyjstat.Dataset.read(response.text)
    return dataset.write('dataframe')


def _clean_raw_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning and filtering to the raw CSO dataset."""
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.replace(' ', '_').str.lower()
    cleaned = cleaned[cleaned['value'] > 0]
    cleaned = cleaned[cleaned['county'] != 'All Counties']
    cleaned = cleaned[cleaned['stamp_duty_event'] != 'Filings']
    return cleaned


def _pivot_price_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Merge mean and median price slices into a single table."""
    df_mean = df[
        (df['statistic'] == 'Mean Sale Price')
        & (df['type_of_buyer'] == 'Household Buyer - First-Time Buyer Owner-Occupier')
    ].rename(columns={'value': 'mean_sale_price'})

    df_median = df[
        (df['statistic'] == 'Median Price')
        & (df['type_of_buyer'] == 'Household Buyer - First-Time Buyer Owner-Occupier')
    ].rename(columns={'value': 'median_price'})

    merged = pd.merge(
        df_mean[['county', 'dwelling_status', 'year', 'mean_sale_price']],
        df_median[['county', 'dwelling_status', 'year', 'median_price']],
        on=['county', 'dwelling_status', 'year'],
        how='inner',
    )
    return merged


def prepare_price_gap_dataset(
    raw_df: pd.DataFrame,
    *,
    min_year: Optional[int] = 2010,
    max_year: Optional[int] = 2023,
) -> PriceGapDataset:
    """Prepare features and targets for modelling the price gap."""
    cleaned = _clean_raw_dataset(raw_df)
    prices = _pivot_price_stats(cleaned)

    prices['gap_ratio'] = prices['mean_sale_price'] / prices['median_price']
    prices['year'] = pd.to_numeric(prices['year'], errors='coerce')

    if min_year is not None:
        prices = prices[prices['year'] >= min_year]
    if max_year is not None:
        prices = prices[prices['year'] <= max_year]

    encoded = pd.get_dummies(prices, columns=['county', 'dwelling_status'], drop_first=True)

    target = encoded['gap_ratio']
    features = encoded.drop(columns=['mean_sale_price', 'median_price', 'gap_ratio'])

    return PriceGapDataset(prices=prices.reset_index(drop=True), features=features, target=target)
