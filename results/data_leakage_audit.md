# Data leakage audit

## Result: FAIL for the claimed independent ISOT-to-WELFake external evaluation

The ISOT train/validation/test split itself passes the exact-content checks: all three pairwise intersections are zero. TF-IDF was fitted on ISOT training data only and the metadata records that WELFake was not used for fitting or tuning.

However, exact normalised content matching finds **39,101 of 39,101 (100.00%)** unique non-empty ISOT articles in raw WELFake. This includes 31,280/31,280 ISOT train, 3,910/3,910 validation and 3,911/3,911 test contents. Consequently, WELFake is not independent of the ISOT training corpus in this workspace.

The published corrected Random Forest WELFake score is therefore reproducible under the documented label mapping, but it **cannot be interpreted as a valid independent cross-dataset generalisation result**. No classifier was rerun or altered in this Stage 1 audit. The earlier opposite-label result remains invalidated.

| Check | Status | Evidence |
|---|---|---|
| ISOT split exact-content leakage | PASS | `tables/duplicate_analysis.csv` |
| TF-IDF fitting leakage | PASS | `baseline_metadata.json` (`ISOT_train_only`) |
| WELFake used for fitting/tuning | PASS | `baseline_metadata.json` (false) |
| ISOT/WELFake external-set independence | FAIL | `tables/dataset_overlap.csv` |
| WELFake external-result generalisation claim | INVALID | Dependent external corpus |
