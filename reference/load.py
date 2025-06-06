import json
import pandas as pd
from collections import defaultdict
from skyfield.api import Loader
from skyfield.data import hipparcos

def stars(magnitude=7.0):
    # load star dataframe & limit magnitude
    load = Loader('./reference/data')
    with load.open(hipparcos.URL) as f:
        df = hipparcos.load_dataframe(f)
        bright_stars_df = df[df['magnitude'] <= magnitude]

    # validate
    if df.empty:
        raise ValueError("Hipparcos data not loaded.")
    elif bright_stars_df.empty:
        raise ValueError(f"No stars found with magnitude <= {magnitude}")

    return bright_stars_df

def planets():
    # load planets
    load = Loader('./reference/data')
    planets = load('de421.bsp')

    # validate
    if planets is None:
        raise ValueError("Planetary data incorrectly loaded.")
    elif 'earth' not in planets:
        raise ValueError("Earth not found in planetary data.")

    return planets

def constellations():
    # load constellation data
    load = Loader('./reference/data')
    url = 'https://raw.githubusercontent.com/Stellarium/stellarium/refs/heads/master/skycultures/modern_st/index.json'

    with load.open(url, filename='constellations.json') as f:
        data = json.load(f)
        const_data = data['constellations']

        const_dict = {const['common_name']['native']: const['lines'] for const in const_data}

    if len(const_dict) == 0 or len(const_dict['Andromeda']) == 0:
        raise ValueError("Constellation data not loaded or is empty.")

    return const_dict
