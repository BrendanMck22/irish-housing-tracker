# import requests
# import pandas as pd
# from pyjstat import pyjstat
# from ydata_profiling import ProfileReport
# import pkg_resources
import requests
import pandas as pd
from pyjstat import pyjstat
import os

url = "https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/HPA02/JSON-stat/2.0/en"

def fetch_and_preprocess_data(save_to: str | None = None) -> pd.DataFrame:

    response = requests.get(url)
    response.raise_for_status()  # Check for errors

    dataset = pyjstat.Dataset.read(response.text)

    df = dataset.write('dataframe')
    df.head()
    # --- STEP 4: Clean column names (optional) ---
    df.columns = df.columns.str.replace(' ', '_').str.lower()
    # remove any rows where value is less than or equal to 0
    df = df[df['value'] > 50000]
    # remove any county that says 'all_counties'
    df = df[df['county'] != 'All Counties']
    # remove any stamp_duy_event that says 'Filings'
    df = df[df['stamp_duty_event'] != 'Filings']
    # remove any dwelling_status that says 'All Dwellings'
    df = df[df['dwelling_status'] != 'All Dwelling Statuses']
    # print unique values in column 'type_of_buyer'
    # Filter for Mean and Median Sale Price
    df_mean = df[(df['statistic'] == 'Mean Sale Price') & (df['type_of_buyer'] == 'Household Buyer - First-Time Buyer Owner-Occupier')]
    df_median = df[(df['statistic'] == 'Median Price') & (df['type_of_buyer'] == 'Household Buyer - First-Time Buyer Owner-Occupier')]
    # # Rename for clarity
    df_mean = df_mean.rename(columns={'value': 'mean_sale_price'})
    df_median = df_median.rename(columns={'value': 'median_price'})

    # Merge on region, dwelling, and quarter
    df_prices = pd.merge(
        df_mean[['county', 'dwelling_status', 'year', 'mean_sale_price']],
        df_median[['county', 'dwelling_status', 'year', 'median_price']],
        on=['county', 'dwelling_status', 'year'],
        how='inner'
    )

    df_prices['gap_ratio'] = df_prices['mean_sale_price'] / df_prices['median_price']
    df_prices['year'] = pd.to_numeric(df_prices['year'], errors='coerce')
    df_prices = df_prices[(df_prices['year'] >= 2010) & (df_prices['year'] <= df_prices['year'].max())]
    return df_prices


