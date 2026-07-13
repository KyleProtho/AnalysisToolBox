# TODO — Statistical Learning Coverage

Tracks coverage against *An Introduction to Statistical Learning* (ISL) chapters.

---

## Chapter 2 — Statistical Learning Basics & KNN

| Status | Function | Module |
|---|---|---|
| ✅ Done | — | — |
| ☐ Needed | `CreateKNearestNeighborModel` | `predictive_analytics/` |

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

## Summary

| | Count |
|---|---|
| ✅ Done | 13 |
| ☐ Needed | 26 |
| **Total** | **39** |

