# Model provenance

The supplied analysis scripts fit a RandomForestRegressor during the
visualization stage using `Row_Spacing` and `Plant_Spacing` as predictors and
`Yield_Score` as the response.

The original scripts do not serialize the transient fitted estimator.
Therefore, the review-package model files were reconstructed from the archived
cultivar-specific simulation-output CSV files using the same model architecture
(`n_estimators=300`, `max_depth=15`).

A fixed `random_state=42` was added only when freezing the review models so that
the serialized files and reviewer predictions are deterministic.

For the supplied archived candidate-summary files (DN251, DN252, and DN253),
the frozen models reproduce the same 100×100-grid virtual 2D candidate
coordinates and the same best plant spacing on the fixed 50-cm row-spacing
transect. Small differences in training R² can occur because the original
transient random forest did not specify a random seed.

See:
`models/reconstruction_check_against_archived_summaries.csv`
