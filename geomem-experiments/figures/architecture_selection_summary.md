# Architecture Selection Summary

Threshold is mean multi-goal optimal-action accuracy under the all-state lookup diagnostic.

| Environment | Candidate set | Threshold | Selected model | Cost | Accuracy | Gain over vector |
|---|---|---:|---|---:|---:|---:|
| trace_commute_000 | conservative_counts | 0.95 | history_counts_h2 | 6 | 1.000 | 0.482 |
| trace_commute_000 | conservative_counts | 0.99 | history_counts_h2 | 6 | 1.000 | 0.482 |
| trace_commute_000 | hybrid_allowed | 0.95 | hybrid_counts_depth_last_h1 | 5 | 1.000 | 0.482 |
| trace_commute_000 | hybrid_allowed | 0.99 | hybrid_counts_depth_last_h1 | 5 | 1.000 | 0.482 |
| trace_commute_033 | conservative_counts | 0.95 | history_counts_h4 | 12 | 0.991 | 0.418 |
| trace_commute_033 | conservative_counts | 0.99 | history_counts_h4 | 12 | 0.991 | 0.418 |
| trace_commute_033 | hybrid_allowed | 0.95 | hybrid_counts_depth_last_h2 | 7 | 0.977 | 0.404 |
| trace_commute_033 | hybrid_allowed | 0.99 | hybrid_counts_depth_last_h4 | 11 | 0.996 | 0.424 |
| trace_commute_067 | conservative_counts | 0.95 | history_counts_h2 | 6 | 0.997 | 0.174 |
| trace_commute_067 | conservative_counts | 0.99 | history_counts_h2 | 6 | 0.997 | 0.174 |
| trace_commute_067 | hybrid_allowed | 0.95 | hybrid_counts_depth_last_h1 | 5 | 0.997 | 0.174 |
| trace_commute_067 | hybrid_allowed | 0.99 | hybrid_counts_depth_last_h1 | 5 | 0.997 | 0.174 |
| trace_commute_100 | conservative_counts | 0.95 | memoryless_counts | 3 | 1.000 | 0.000 |
| trace_commute_100 | conservative_counts | 0.99 | memoryless_counts | 3 | 1.000 | 0.000 |
| trace_commute_100 | hybrid_allowed | 0.95 | memoryless_counts | 3 | 1.000 | 0.000 |
| trace_commute_100 | hybrid_allowed | 0.99 | memoryless_counts | 3 | 1.000 | 0.000 |
