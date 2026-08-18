# Load packages
import re
import pandas as pd
import numpy as np

# Module-level cache for geoBoundaries API responses. Keyed by (iso3, admin_level),
# this persists for the life of the Python session/notebook so that calling this
# function repeatedly (e.g. across cells) does not re-hit the API for boundaries
# already fetched.
_GEOBOUNDARIES_CACHE = {}

# U.S. state names and postal abbreviations, used both for geography_level='auto'
# inference and to keep the module self-contained (no extra dependency for a
# lookup this small).
_US_STATES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
    'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
    'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
    'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
    'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
    'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia',
}

_VALID_GEOGRAPHY_LEVELS = ['country', 'us_state', 'us_county', 'zip', 'admin1', 'admin2']


# Declare function
def CreateChoroplethMapFromNames(dataframe,
                                 location_column,
                                 value_column,
                                 geography_level='auto',
                                 country_column=None,
                                 boundary_source='geoboundaries',
                                 api_key=None,
                                 fuzzy_match_threshold=85,
                                 normalize_diacritics=True,
                                 census_year=2021,
                                 map_choropleth=False):
    """
    Resolve free-text place names to polygons and build a choropleth map.

    This function fills the gap between "I have a spreadsheet column of place names" and
    "I have a map." Analysts frequently receive data with a geography column that contains
    only names — country names, U.S. state or county names, ZIP codes, or subnational
    international divisions — rather than shapefiles, latitude/longitude pairs, or FIPS
    codes. This function resolves those names to boundary polygons automatically, in the
    spirit of Excel's "Geography" data type / Map Chart feature: give it names, and it
    figures out the shapes. It normalizes and fuzzy-matches names against an authoritative
    boundary source (the U.S. Census Bureau's TIGER database or the geoBoundaries project),
    attaches match confidence scores, and optionally renders an interactive Folium
    choropleth colored by a value column.

    Resolving named geographies to boundaries is essential for:
      * Turning open-source reporting (news articles, sanctions lists, incident logs) that
        reference place names into mappable intelligence products
      * Rapidly visualizing survey, sales, or population data that was collected by
        place name rather than by geographic identifier
      * Cross-referencing OSINT datasets from different sources that describe the same
        geography inconsistently (e.g. "USA" vs "United States" vs "U.S.")
      * Building situational-awareness maps during fast-moving events, where analysts are
        working from name-based source material, not GIS-ready data
      * Quality-checking name-based geographic fields before they are joined to other
        spatial datasets, by surfacing unmatched or ambiguous names rather than dropping them
      * Supporting country- and subnational-level trend analysis without requiring analysts
        to hand-source or maintain their own shapefiles

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataset containing a column of place names and a column of values to map.
    location_column : str
        The name of the column in `dataframe` containing place names to resolve (e.g.
        country names, U.S. state/county names, ZIP codes, or subnational division names).
    value_column : str
        The name of the numeric column to color the choropleth by (e.g. counts, rates,
        scores). Also carried through to the returned matched GeoDataFrame.
    geography_level : str, optional
        The type of geography represented by `location_column`. One of 'auto', 'country',
        'us_state', 'us_county', 'zip', 'admin1' (first-level subnational, e.g. states or
        provinces outside the U.S.), or 'admin2' (second-level subnational, e.g. districts
        or counties outside the U.S.). Defaults to 'auto', which inspects the values in
        `location_column` and infers the level. If the values are too ambiguous to infer
        confidently (which is common for admin1/admin2 names), a ValueError is raised asking
        the caller to specify `geography_level` explicitly.
    country_column : str, optional
        The name of a column in `dataframe` containing the country each row belongs to.
        When provided, it restricts fuzzy-matching candidates to boundaries within that
        country, which reduces false positives on ambiguous names (e.g. a district name
        that exists in multiple countries). Required when `boundary_source='geoboundaries'`
        and `geography_level` is 'admin1' or 'admin2', since the geoBoundaries API is
        queried per country. Defaults to None.
    boundary_source : str, optional
        Where to source boundary polygons from. 'geoboundaries' (default) fetches boundaries
        from the geoBoundaries project's API and is licensed CC-BY, which makes it safe for
        commercial and redistributable use (unlike GADM, whose license restricts
        redistribution) — this is why it is the default rather than GADM. 'census' fetches
        boundaries from the U.S. Census Bureau's TIGER database via the existing
        `FetchUSShapefile` function and only supports `geography_level` values 'us_state',
        'us_county', and 'zip'. Defaults to 'geoboundaries'.
    api_key : str, optional
        An API key to send as an `Authorization` header on outbound requests to
        `boundary_source`'s API. geoBoundaries' official public API
        (https://www.geoboundaries.org) is free and does not require a key as of this
        function's implementation, so this can be left as None for the default path. It
        exists for forward compatibility with rate-limited mirrors, proxies, or alternative
        keyed boundary providers. Defaults to None.
    fuzzy_match_threshold : int, optional
        The minimum rapidfuzz token-sort-ratio score (0-100) required to accept a fuzzy
        match when an exact match isn't found. Defaults to 85.
    normalize_diacritics : bool, optional
        Whether to strip accents/diacritics (e.g. "Cordoba" vs "Córdoba") when comparing
        names. The original boundary name is always returned unmodified; stripping is only
        used to improve matching. Defaults to True.
    census_year : int, optional
        The census year to request when `boundary_source='census'`. Passed through to
        `FetchUSShapefile`. Defaults to 2021.
    map_choropleth : bool, optional
        Whether to generate and display an interactive Folium choropleth map of the matched
        geographies, colored by `value_column`. Defaults to False.

    Returns
    -------
    tuple of (geopandas.GeoDataFrame, pd.DataFrame)
        matched_geodataframe : geopandas.GeoDataFrame
            The original dataframe joined with polygon `geometry`, `matched_boundary_name`
            (the original, un-normalized name of the boundary that was matched), and
            `match_confidence` (100 for an exact match, otherwise the fuzzy match score).
        unmatched_names_df : pd.DataFrame
            The subset of rows whose `location_column` value could not be resolved at or
            above `fuzzy_match_threshold`, including a `best_fuzzy_score` column, so the
            analyst can inspect and fix them rather than have them silently dropped.

    Examples
    --------
    # OSINT: Mapping incident counts by country (geoBoundaries, CC-BY licensed, default)
    import pandas as pd
    incident_data = pd.DataFrame({
        'country': ['Kenya', 'Boliva', 'Cote d\\'Ivoire', 'Not A Real Country'],
        'incident_count': [12, 4, 7, 1]
    })
    matched, unmatched = CreateChoroplethMapFromNames(
        incident_data,
        location_column='country',
        value_column='incident_count',
        geography_level='country',
        boundary_source='geoboundaries',
        map_choropleth=True
    )
    # `matched` has one polygon per resolved country; 'Not A Real Country' lands in `unmatched`

    # U.S. state-level values from the Census Bureau's TIGER shapefiles
    state_data = pd.DataFrame({
        'state': ['California', 'Texas', 'New York'],
        'sales': [500000, 420000, 610000]
    })
    matched, unmatched = CreateChoroplethMapFromNames(
        state_data,
        location_column='state',
        value_column='sales',
        geography_level='us_state',
        boundary_source='census',
        census_year=2021
    )
    """
    # Lazy load uncommon packages
    import geopandas as gpd
    import requests
    from unidecode import unidecode
    from rapidfuzz import fuzz, process
    import pycountry

    # Validate inputs
    if location_column not in dataframe.columns:
        raise ValueError(
            f"location_column '{location_column}' was not found in the dataframe. "
            f"Available columns: {list(dataframe.columns)}"
        )
    if value_column not in dataframe.columns:
        raise ValueError(
            f"value_column '{value_column}' was not found in the dataframe. "
            f"Available columns: {list(dataframe.columns)}"
        )
    if country_column is not None and country_column not in dataframe.columns:
        raise ValueError(
            f"country_column '{country_column}' was not found in the dataframe. "
            f"Available columns: {list(dataframe.columns)}"
        )
    if boundary_source not in ('geoboundaries', 'census'):
        raise ValueError("boundary_source must be one of 'geoboundaries' or 'census'.")
    if geography_level != 'auto' and geography_level not in _VALID_GEOGRAPHY_LEVELS:
        raise ValueError(
            f"geography_level must be 'auto' or one of {_VALID_GEOGRAPHY_LEVELS}, "
            f"got '{geography_level}'."
        )

    df = dataframe.copy()

    # Infer geography_level if requested
    if geography_level == 'auto':
        geography_level = _InferGeographyLevel(df[location_column])

    def normalize(text):
        if pd.isna(text):
            return ''
        cleaned = re.sub(r'\s+', ' ', str(text).strip())
        if normalize_diacritics:
            cleaned = unidecode(cleaned)
        return cleaned.lower()

    # Resolve boundaries and run the matching logic. Country-level matches sourced from
    # geoBoundaries are handled separately because they are resolved primarily via
    # pycountry rather than fuzzy-matching against a fetched name field (see docstring
    # and BEHAVIOR notes in the geoBoundaries fetch helper below).
    if boundary_source == 'geoboundaries' and geography_level == 'country':
        matched_records, unmatched_records = _MatchCountriesViaPycountry(
            df=df,
            location_column=location_column,
            normalize=normalize,
            fuzzy_match_threshold=fuzzy_match_threshold,
            api_key=api_key,
            requests_module=requests,
            gpd=gpd,
            pycountry=pycountry,
            fuzz=fuzz,
        )
    else:
        if boundary_source == 'census':
            boundary_gdf, name_field = _FetchCensusBoundaries(geography_level, census_year)
        else:
            if country_column is None:
                raise ValueError(
                    "country_column is required when boundary_source='geoboundaries' and "
                    "geography_level is 'admin1' or 'admin2', since the geoBoundaries API "
                    "is queried per country. Please provide country_column."
                )
            boundary_gdf, name_field = _FetchGeoBoundariesAdmin(
                df=df,
                country_column=country_column,
                geography_level=geography_level,
                api_key=api_key,
                requests_module=requests,
                gpd=gpd,
                pycountry=pycountry,
            )

        iso3_lookup = None
        if country_column is not None and 'iso3' in boundary_gdf.columns:
            iso3_lookup = {
                value: _NormalizeCountryToISO3(value, pycountry)
                for value in df[country_column].dropna().unique()
            }

        matched_records, unmatched_records = _MatchAgainstBoundaries(
            df=df,
            location_column=location_column,
            country_column=country_column,
            boundary_gdf=boundary_gdf,
            name_field=name_field,
            normalize=normalize,
            fuzzy_match_threshold=fuzzy_match_threshold,
            fuzz=fuzz,
            process=process,
            iso3_lookup=iso3_lookup,
        )

    matched_df = pd.DataFrame(matched_records)
    if len(matched_df) > 0:
        matched_gdf = gpd.GeoDataFrame(matched_df, geometry='geometry', crs='EPSG:4326')
    else:
        matched_gdf = gpd.GeoDataFrame(matched_df, geometry=[], crs='EPSG:4326')

    unmatched_names_df = pd.DataFrame(unmatched_records)

    if map_choropleth and len(matched_gdf) > 0:
        _RenderChoroplethMap(matched_gdf, value_column)

    return matched_gdf, unmatched_names_df


def _InferGeographyLevel(location_series):
    """Infer geography_level from the values in location_column, or raise ValueError."""
    import pycountry

    values = location_series.dropna().astype(str).str.strip()
    values = values[values != '']
    if len(values) == 0:
        raise ValueError(
            "Could not infer geography_level because location_column contains no "
            "non-null values. Please specify geography_level explicitly as one of: "
            f"{_VALID_GEOGRAPHY_LEVELS}."
        )

    # Cap the sample for performance on large dataframes
    sample = values.unique()[:200]
    n = len(sample)

    # ZIP codes: 5-digit, optionally ZIP+4
    zip_pattern = re.compile(r'^\d{5}(-\d{4})?$')
    zip_hits = sum(1 for v in sample if zip_pattern.match(v))
    if zip_hits / n >= 0.9:
        return 'zip'

    # Countries, via pycountry
    country_hits = 0
    for v in sample:
        try:
            pycountry.countries.lookup(v)
            country_hits += 1
        except LookupError:
            continue
    if country_hits / n >= 0.8:
        return 'country'

    # U.S. states, by name or postal abbreviation
    state_names_lower = {name.lower() for name in _US_STATES.values()}
    state_abbrs_lower = {abbr.lower() for abbr in _US_STATES}
    state_hits = sum(
        1 for v in sample if v.lower() in state_names_lower or v.lower() in state_abbrs_lower
    )
    if state_hits / n >= 0.8:
        return 'us_state'

    # U.S. counties commonly carry a recognizable suffix
    county_suffix_pattern = re.compile(r'\b(county|parish|borough|census area)\b', re.IGNORECASE)
    county_hits = sum(1 for v in sample if county_suffix_pattern.search(v))
    if county_hits / n >= 0.6:
        return 'us_county'

    raise ValueError(
        "Could not confidently infer geography_level from location_column values "
        f"(sample: {list(sample[:5])}). Please specify geography_level explicitly as "
        f"one of: {_VALID_GEOGRAPHY_LEVELS}. Note that 'admin1' and 'admin2' (subnational "
        "divisions outside the U.S.) cannot be auto-inferred and must always be specified "
        "explicitly."
    )


def _NormalizeCountryToISO3(value, pycountry):
    """Resolve free-text country name/code to an ISO 3166-1 alpha-3 code, or None."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return pycountry.countries.lookup(text).alpha_3
    except LookupError:
        pass
    try:
        results = pycountry.countries.search_fuzzy(text)
        return results[0].alpha_3 if results else None
    except LookupError:
        return None


def _FetchCensusBoundaries(geography_level, census_year):
    """Fetch boundary polygons for a U.S. geography level via FetchUSShapefile."""
    from analysistoolbox.data_collection.FetchUSShapefile import FetchUSShapefile

    geography_map = {
        'us_state': 'states',
        'us_county': 'counties',
        'zip': 'zipcodes',
    }
    if geography_level not in geography_map:
        raise ValueError(
            f"boundary_source='census' only supports geography_level values "
            f"{list(geography_map.keys())}, got '{geography_level}'."
        )
    geography = geography_map[geography_level]

    try:
        boundary_gdf = FetchUSShapefile(state=None, geography=geography, census_year=census_year)
    except Exception as e:
        raise ValueError(
            f"Failed to fetch U.S. Census '{geography}' shapefile for "
            f"census_year={census_year}: {e}"
        ) from e

    if geography_level == 'zip':
        preferred_names = [c for c in boundary_gdf.columns if str(c).upper().startswith('ZCTA5CE')]
    else:
        preferred_names = ['NAME']

    name_field = next((c for c in preferred_names if c in boundary_gdf.columns), None)
    if name_field is None:
        name_field = next(
            (c for c in boundary_gdf.columns if c != 'geometry' and boundary_gdf[c].dtype == object),
            None,
        )
    if name_field is None:
        raise ValueError(
            f"Could not determine a name field on the fetched '{geography}' shapefile. "
            f"Available columns: {list(boundary_gdf.columns)}"
        )

    return boundary_gdf, name_field


def _FetchGeoBoundariesLevel(iso3, admin_level, api_key, requests_module, gpd):
    """
    Fetch a single country + admin-level boundary set from the geoBoundaries API,
    caching the result in-session.

    geoBoundaries' official API (https://www.geoboundaries.org/api.html) is free and does
    not require an API key as of this function's implementation. The endpoint used here,
    https://www.geoboundaries.org/api/current/gbOpen/{ISO3}/{ADM_LEVEL}/, returns JSON
    metadata (including a `gjDownloadURL` pointing at the actual GeoJSON boundary file) for
    the gbOpen (CC-BY licensed) product. api_key is accepted purely for forward
    compatibility with rate-limited mirrors or alternative keyed providers -- it is sent as
    an Authorization header when supplied, but is not required for the public API.
    """
    cache_key = (iso3, admin_level)
    if cache_key in _GEOBOUNDARIES_CACHE:
        return _GEOBOUNDARIES_CACHE[cache_key]

    headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
    metadata_url = f'https://www.geoboundaries.org/api/current/gbOpen/{iso3}/{admin_level}/'

    try:
        metadata_response = requests_module.get(metadata_url, headers=headers, timeout=30)
        metadata_response.raise_for_status()
        metadata = metadata_response.json()
        download_url = metadata['gjDownloadURL']

        geojson_response = requests_module.get(download_url, headers=headers, timeout=60)
        geojson_response.raise_for_status()
        geojson = geojson_response.json()

        boundary_gdf = gpd.GeoDataFrame.from_features(geojson['features'])
        boundary_gdf = boundary_gdf.set_crs('EPSG:4326', allow_override=True)
    except Exception as e:
        raise ValueError(
            f"Failed to fetch geoBoundaries data for country '{iso3}' at admin level "
            f"'{admin_level}': {e}"
        ) from e

    _GEOBOUNDARIES_CACHE[cache_key] = boundary_gdf
    return boundary_gdf


def _FetchGeoBoundariesAdmin(df, country_column, geography_level, api_key, requests_module, gpd, pycountry):
    """Fetch admin1/admin2 boundaries for every country referenced in country_column."""
    admin_level = 'ADM1' if geography_level == 'admin1' else 'ADM2'

    countries = df[country_column].dropna().unique()
    if len(countries) == 0:
        raise ValueError(f"country_column '{country_column}' contains no non-null values.")

    boundary_frames = []
    for country_value in countries:
        iso3 = _NormalizeCountryToISO3(country_value, pycountry)
        if iso3 is None:
            raise ValueError(
                f"Could not resolve country '{country_value}' in country_column "
                f"'{country_column}' to an ISO 3166-1 country code."
            )
        country_gdf = _FetchGeoBoundariesLevel(iso3, admin_level, api_key, requests_module, gpd)
        country_gdf = country_gdf.copy()
        country_gdf['iso3'] = iso3
        boundary_frames.append(country_gdf)

    boundary_gdf = gpd.GeoDataFrame(pd.concat(boundary_frames, ignore_index=True), crs='EPSG:4326')

    name_field = 'shapeName' if 'shapeName' in boundary_gdf.columns else next(
        (c for c in boundary_gdf.columns if c not in ('geometry', 'iso3') and boundary_gdf[c].dtype == object),
        None,
    )
    if name_field is None:
        raise ValueError(
            f"Could not determine a name field on the fetched geoBoundaries '{admin_level}' data. "
            f"Available columns: {list(boundary_gdf.columns)}"
        )

    return boundary_gdf, name_field


def _MatchAgainstBoundaries(df, location_column, country_column, boundary_gdf, name_field,
                            normalize, fuzzy_match_threshold, fuzz, process, iso3_lookup):
    """Exact-then-fuzzy match each row's location_column value against boundary_gdf[name_field]."""
    boundary_gdf = boundary_gdf.copy()
    boundary_gdf['_norm_name'] = boundary_gdf[name_field].apply(normalize)

    matched_records = []
    unmatched_records = []

    for _, row in df.iterrows():
        name_norm = normalize(row[location_column])

        candidates = boundary_gdf
        if country_column is not None and iso3_lookup is not None and 'iso3' in boundary_gdf.columns:
            country_iso3 = iso3_lookup.get(row[country_column])
            if country_iso3:
                subset = boundary_gdf[boundary_gdf['iso3'] == country_iso3]
                if len(subset) > 0:
                    candidates = subset

        exact_matches = candidates[candidates['_norm_name'] == name_norm]
        if len(exact_matches) > 0:
            best_row = exact_matches.iloc[0]
            confidence = 100
        else:
            choices = candidates['_norm_name'].tolist()
            result = process.extractOne(name_norm, choices, scorer=fuzz.token_sort_ratio) if choices else None
            if result is not None and result[1] >= fuzzy_match_threshold:
                best_row = candidates.iloc[result[2]]
                confidence = result[1]
            else:
                best_row = None
                confidence = result[1] if result is not None else 0

        if best_row is not None:
            record = row.to_dict()
            record['geometry'] = best_row['geometry']
            record['matched_boundary_name'] = best_row[name_field]
            record['match_confidence'] = confidence
            matched_records.append(record)
        else:
            record = row.to_dict()
            record['best_fuzzy_score'] = confidence
            unmatched_records.append(record)

    return matched_records, unmatched_records


def _MatchCountriesViaPycountry(df, location_column, normalize, fuzzy_match_threshold,
                                api_key, requests_module, gpd, pycountry, fuzz):
    """
    Resolve country names via pycountry first, then fetch ADM0 boundary geometry from
    geoBoundaries for each resolved ISO3 code. This handles the vast majority of country
    names without needing fuzzy-matching against boundary names directly.
    """
    matched_records = []
    unmatched_records = []

    for _, row in df.iterrows():
        raw_value = row[location_column]
        name_norm = normalize(raw_value)
        if not name_norm:
            record = row.to_dict()
            record['best_fuzzy_score'] = 0
            unmatched_records.append(record)
            continue

        confidence = 0
        resolved_country = None
        try:
            resolved_country = pycountry.countries.lookup(str(raw_value).strip())
            confidence = 100
        except LookupError:
            try:
                fuzzy_results = pycountry.countries.search_fuzzy(str(raw_value).strip())
            except LookupError:
                fuzzy_results = []
            if fuzzy_results:
                candidate = fuzzy_results[0]
                score = fuzz.token_sort_ratio(name_norm, normalize(candidate.name))
                if score >= fuzzy_match_threshold:
                    resolved_country = candidate
                    confidence = score
                else:
                    confidence = score

        if resolved_country is None:
            record = row.to_dict()
            record['best_fuzzy_score'] = confidence
            unmatched_records.append(record)
            continue

        try:
            country_gdf = _FetchGeoBoundariesLevel(
                resolved_country.alpha_3, 'ADM0', api_key, requests_module, gpd
            )
        except ValueError as e:
            raise ValueError(
                f"Resolved '{raw_value}' to country '{resolved_country.name}' "
                f"({resolved_country.alpha_3}), but failed to fetch its boundary: {e}"
            ) from e

        if len(country_gdf) == 0:
            record = row.to_dict()
            record['best_fuzzy_score'] = confidence
            unmatched_records.append(record)
            continue

        boundary_row = country_gdf.iloc[0]
        matched_name = boundary_row['shapeName'] if 'shapeName' in country_gdf.columns else resolved_country.name

        record = row.to_dict()
        record['geometry'] = boundary_row['geometry']
        record['matched_boundary_name'] = matched_name
        record['match_confidence'] = confidence
        matched_records.append(record)

    return matched_records, unmatched_records


def _RenderChoroplethMap(matched_gdf, value_column):
    """Render an interactive Folium choropleth colored by value_column."""
    import folium
    from IPython.display import display

    map_gdf = matched_gdf.reset_index(drop=True).copy()
    map_gdf['_row_id'] = map_gdf.index.astype(str)

    centroid = map_gdf.geometry.unary_union.centroid
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=3)

    folium.Choropleth(
        geo_data=map_gdf[['_row_id', 'geometry']].to_json(),
        data=map_gdf,
        columns=['_row_id', value_column],
        key_on='feature.properties._row_id',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.3,
        legend_name=value_column,
    ).add_to(m)

    tooltip_fields = ['matched_boundary_name', value_column]
    folium.GeoJson(
        map_gdf,
        style_function=lambda feature: {'fillOpacity': 0, 'weight': 0},
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields),
    ).add_to(m)

    try:
        display(m)
    except Exception:
        pass  # Not in IPython/Jupyter

    return m
