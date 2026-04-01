import os
import pandas as pd
from constants import DATE_COL, ZONE_ID_COL

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "1_data", "processed", "zone_sequence_merged.csv")


def load_zone_data(zone_id):
    """Load data for a single zone."""
    df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COL])
    df = df[df[ZONE_ID_COL] == zone_id]
    df = df.drop(columns=[ZONE_ID_COL])
    df = df.sort_values(by=DATE_COL).reset_index(drop=True)
    return df


def load_regional_data():
    """Load and aggregate data across all zones into a single regional series."""
    df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COL])
    df = df.drop(columns=[ZONE_ID_COL])
    df = df.groupby(DATE_COL).agg({
        'Precipitation': 'mean',
        'Humidity': 'mean',
        'Temperature': 'mean',
        'Wind': 'mean',
        'Num_Fires': 'sum'
    }).reset_index()
    df = df.sort_values(by=DATE_COL).reset_index(drop=True)
    return df