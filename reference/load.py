from skyfield.api import Loader, load
from skyfield.data import hipparcos, stellarium

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
    with load.open('./reference/data/constellationship.fab') as f:
        constellations = stellarium.parse_constellations(f)

    return constellations
