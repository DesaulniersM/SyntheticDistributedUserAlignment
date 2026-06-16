Bookmarked: High-Volume Statistical Study -> Run 30-50 trials per cell to generate Confidence Intervals and Probability Density Functions for solver convergence.

## Certificate-Aware Subgraph Merging (2026 Research Direction)
- **Concept**: Use SE-Sync's global optimality certificate (eigenvalue gap/duality gap) as a "Quality Gate" before merging disconnected swarms.
- **Workflow**: Subgraphs calculate their own consensus scores. A merge is only initiated if both subgraphs are internally consistent. If one is "mushy" (low consensus), the merge is delayed or the bad subgraph is down-weighted to prevent global map corruption.
- **Advantage over GTSAM**: Provides a diagnostic "Score" as a byproduct of the solve, enabling predictive rather than reactive map management.
