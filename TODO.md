# TODO — Statistical Learning Coverage

Tracks coverage against *An Introduction to Statistical Learning* (ISL) chapters.

---

## Chapter 2 — Statistical Learning Basics & KNN

| Status | Function | Module |
|---|---|---|
| ✅ Done | `CreateKNearestNeighborModel` | `predictive_analytics/` |

---

## Chapter 3 — Linear Regression

| Status | Function | Module |
|---|---|---|
| ✅ Done | `ConductLinearRegressionAnalysis` | `hypothesis_testing/` |
| ✅ Done | `CreateLinearRegressionModel` | `predictive_analytics/` |

---

## Chapter 4 — Classification

| Status | Function | Module |
|---|---|---|
| ✅ Done | `ConductLogisticRegressionAnalysis` | `hypothesis_testing/` |
| ✅ Done | `CreateLogisticRegressionModel` | `predictive_analytics/` |
| ☐ Needed | `ConductLinearDiscriminantAnalysis` | `hypothesis_testing/` |

---

## Chapter 5 — Resampling Methods

| Status | Function | Module |
|---|---|---|
| ☐ Needed | `ConductCrossValidation` | `predictive_analytics/` |
| ☐ Needed | `ConductBootstrapAnalysis` | `statistics/` |

---

## Chapter 6 — Linear Model Selection & Regularization

| Status | Function | Module |
|---|---|---|
| ☐ Needed | `ConductStepwiseSelection` | `predictive_analytics/` |
| ☐ Needed | `CreateRidgeRegressionModel` | `predictive_analytics/` |
| ☐ Needed | `CreateLassoRegressionModel` | `predictive_analytics/` |
| ☐ Needed | `CreateElasticNetRegressionModel` | `predictive_analytics/` |
| ☐ Needed | `CreatePrincipalComponentsRegressionModel` | `predictive_analytics/` |

---

## Chapter 7 — Moving Beyond Linearity

| Status | Function | Module |
|---|---|---|
| ☐ Needed | `CreatePolynomialRegressionModel` | `predictive_analytics/` |
| ☐ Needed | `CreateSplineRegressionModel` | `predictive_analytics/` |
| ☐ Needed | `CreateGeneralizedAdditiveModel` | `predictive_analytics/` |

---

## Chapter 8 — Tree-Based Methods

| Status | Function | Module |
|---|---|---|
| ✅ Done | `CreateDecisionTreeModel` | `predictive_analytics/` |
| ✅ Done | `CreateBoostedTreeModel` | `predictive_analytics/` |
| ☐ Needed | `CreateBaggingModel` | `predictive_analytics/` |
| ☐ Needed | `CreateRandomForestModel` | `predictive_analytics/` |

---

## Chapter 9 — Support Vector Machines

| Status | Function | Module |
|---|---|---|
| ☐ Needed | `CreateSupportVectorMachineModel` | `predictive_analytics/` |

---

## Chapter 10 — Deep Learning

| Status | Function | Module |
|---|---|---|
| ✅ Done | `CreateNeuralNetwork_SingleOutcome` | `predictive_analytics/` |

---

## Chapter 11 — Survival Analysis

| Status | Function | Module |
|---|---|---|
| ✅ Done | `ConductSurvivalAnalysis` | `hypothesis_testing/` |
| ✅ Done | `ConductCoxProportionalHazardRegression` | `hypothesis_testing/` |

---

## Chapter 12 — Unsupervised Learning

| Status | Function | Module |
|---|---|---|
| ✅ Done | `ConductPrincipalComponentAnalysis` | `descriptive_analytics/` |
| ✅ Done | `CreateKMeansClusters` | `descriptive_analytics/` |
| ✅ Done | `CreateHierarchicalClusters` | `descriptive_analytics/` |

---

## Chapter 13 — Multiple Hypothesis Testing

| Status | Function | Module |
|---|---|---|
| ☐ Needed | `ConductMultipleHypothesisTesting` | `hypothesis_testing/` |

---

## OSINT Data Collection

Functions for gathering open-source intelligence from publicly available sources.
Scoped to public APIs and legally accessible data only — excludes anything that primarily enables individual surveillance (people-search engines, phone lookups, social scraping, criminal/voter records).

| Status | Function | Module | Source / Notes |
|---|---|---|---|
| ☐ Needed | `FetchDNSRecords` | `data_collection/` | DNS lookups (A, MX, NS, TXT, SOA) via `dnspython`; DNS data is fully public |
| ☐ Needed | `FetchWHOISData` | `data_collection/` | WHOIS registration data for domains and IPs via `python-whois` |
| ☐ Needed | `FetchIPGeolocation` | `data_collection/` | Geolocate an IP address using public APIs (ip-api.com / ipinfo.io); returns city/region/ASN, not individual PII |
| ☐ Needed | `FetchWebArchiveSnapshot` | `data_collection/` | Retrieve archived page snapshots from the Internet Archive Wayback Machine CDX API |
| ☐ Needed | `FetchPatentRecords` | `data_collection/` | Search USPTO open API or Google Patents by keyword, assignee, or CPC class |
| ☐ Needed | `FetchWorldBankIndicator` | `data_collection/` | Download economic/development indicators from the World Bank Open Data API |
| ☐ Needed | `FetchGovernmentDataset` | `data_collection/` | Search and download datasets from Data.gov and similar open government portals |
| ☐ Needed | `ExtractDocumentMetadata` | `data_collection/` | Extract embedded metadata (author, timestamps, GPS coords) from PDFs and images via `pymupdf` / `Pillow` |
| ☐ Needed | `FetchThreatIntelIOC` | `data_collection/` | Check IPs, domains, and file hashes against public threat feeds (AlienVault OTX API); defensive/research use |
| ☐ Needed | `DecodeEncodedString` | `data_processing/` | Detect and decode Base64, hex, URL-encoding, and ROT13; useful for analyzing encoded artifacts |

---

## Geospatial / OSINT Analysis Primitives

Geospatial primitives and higher-level analytic techniques for tabular location data. Wave 1 items are
low-dependency primitives that unlock the rest; Wave 2 items build on them.

### Wave 1 — Primitives

| Status | Function | Module | Notes |
|---|---|---|---|
| ✅ Done | `CalculateHaversineDistance` | `geospatial_analysis/` | Pairwise or point-to-point great-circle distance between rows. `ConductClusterAnalysis` already uses haversine internally — pull it out as a standalone, reusable primitive. Pure numpy, no new dependency. |
| ☐ Needed | `ConvertCoordinateFormats` | `geospatial_analysis/` | Bidirectional conversion between decimal degrees, DMS, UTM, and MGRS on a tabular column. MGRS is the geocoordinate standard used by NATO militaries for geo-referencing and position reporting; analysts working from military reporting, satellite imagery metadata, or European mapping sources need to normalize into one format before analysis. Candidate libs: `mgrs`, `utm`, or `pyproj`. |
| ✅ Done | `FindNearestPointOfInterest` | `geospatial_analysis/` | Given a dataframe of observations and a dataframe of reference points (facilities, checkpoints, prior sightings), return nearest reference point + distance for each row. Uses `sklearn.neighbors.BallTree` with haversine metric — fast at scale, natural companion to the existing cluster function. |

### Wave 2 — Higher-Level Techniques (built on Wave 1)

| Status | Function | Module | Notes |
|---|---|---|---|
| ☐ Needed | `CalculateBearingAndSpeed` | `geospatial_analysis/` | For time-ordered tracks (person, vehicle, vessel, aircraft with sequential lat/lon/timestamp rows), compute bearing/heading and speed between consecutive points. Useful for flagging anomalous movement (loitering, sudden course change, implausible speed) — a common OSINT pattern-of-life technique. |
| ☐ Needed | `CheckPointInPolygon` / `CalculateGeofenceDwellTime` | `geospatial_analysis/` | Flag whether tabular points fall inside a boundary (from `FetchUSShapefile` output or an uploaded GeoJSON/shapefile) and, for time-series data, how long a subject dwelled inside it. Pairs naturally with the existing shapefile fetcher. |
| ☐ Needed | `CalculateConvexHull` | `geospatial_analysis/` | Compute the minimum bounding polygon (and area) around a set of points — useful for defining an "area of interest" or estimating the spatial extent of an entity's activity from scattered observations. `scipy.spatial.ConvexHull`, no heavy new dependency. |
| ☐ Needed | `ReverseGeocode` | `geospatial_analysis/` | Complement to `GeocodeUSAddresses` — lat/lon back to a human-readable address/place name via Nominatim/OSM, so it isn't US-only. Fills the international gap the current geocoder leaves. |
| ☐ Needed | `GenerateKernelDensityHeatmap` | `geospatial_analysis/` | KDE-based density surface (vs. the hard clustering of DBSCAN) rendered as a Folium heatmap — good for "where is activity concentrated" questions where discrete clusters aren't the right framing. |

---

## Summary

| | Count |
|---|---|
| ✅ Done | 15 |
| ☐ Needed | 32 |
| **Total** | **47** |

