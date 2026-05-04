# Graph Report - .  (2026-05-04)

## Corpus Check
- Large corpus: 11 files · ~875,964 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder, or use --no-semantic to run AST-only.

## Summary
- 24 nodes · 20 edges · 5 communities detected
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 2 edges (avg confidence: 0.88)
- Token cost: 23,539 input · 23,539 output

## Community Hubs (Navigation)
- [[_COMMUNITY_RANSAC Edge Detection|RANSAC Edge Detection]]
- [[_COMMUNITY_Force Error Calculation|Force Error Calculation]]
- [[_COMMUNITY_Experimental Data Analysis|Experimental Data Analysis]]
- [[_COMMUNITY_Tool Wear Measurement|Tool Wear Measurement]]
- [[_COMMUNITY_B-Spline Optimization|B-Spline Optimization]]

## God Nodes (most connected - your core abstractions)
1. `computeResiduals_Spline Function` - 4 edges
2. `analyze_single_side_silent Function` - 4 edges
3. `calculateForceError_Fixed Function` - 3 edges
4. `run_Spline_Optimizer Script` - 3 edges
5. `ForceDataAnalysisv11tool15 Script` - 3 edges
6. `ForceDataAnalysisv13tool15 Script` - 3 edges
7. `Hardstava Tool 15 Experimental Force Data` - 3 edges
8. `wornArea Main Function` - 2 edges
9. `calculate_edge_radii Function` - 2 edges
10. `ransac_filter_interval Function` - 2 edges

## Surprising Connections (you probably didn't know these)
- `calculateForceError_Fixed Function` --semantically_similar_to--> `computeResiduals_Spline Function`  [INFERRED] [semantically similar]
  calculateForceError_Fixed.m → computeResiduals_Spline.m
- `run_Spline_Optimizer Script` --references--> `Hardstava Tool 15 Experimental Force Data`  [EXTRACTED]
  run_Spline_Optimizer.m → hardstavaxtool15datalong.txt
- `ForceDataAnalysisv11tool15 Script` --semantically_similar_to--> `ForceDataAnalysisv13tool15 Script`  [INFERRED] [semantically similar]
  ForceDataAnalysisv11tool15.m → ForceDataAnalysisv13tool15.m
- `ForceDataAnalysisv11tool15 Script` --references--> `Hardstava Tool 15 Experimental Force Data`  [EXTRACTED]
  ForceDataAnalysisv11tool15.m → hardstavaxtool15datalong.txt
- `ForceDataAnalysisv13tool15 Script` --references--> `Hardstava Tool 15 Experimental Force Data`  [EXTRACTED]
  ForceDataAnalysisv13tool15.m → hardstavaxtool15datalong.txt

## Hyperedges (group relationships)
- **Force Modeling and Optimization Pipeline** — calculateForceError_Fixed, computeResiduals_Spline, run_Spline_Optimizer, hardstavaxtool15datalong [INFERRED 0.85]
- **Experimental Tooth-Level Force Analysis** — ForceDataAnalysisv11tool15, ForceDataAnalysisv13tool15, hardstavaxtool15datalong [EXTRACTED 1.00]
- **Tool Wear Edge Detection and RANSAC Chain** — wornArea, calculate_edge_radii, analyze_single_side_silent, ransac_filter_interval [EXTRACTED 1.00]

## Communities (11 total, 1 thin omitted)

### Community 0 - "RANSAC Edge Detection"
Cohesion: 0.4
Nodes (5): analyze_single_side_silent Function, circfit_radius_only Function, findRefinedAnchor Function, ransac_filter_interval Function, run_ransac_fixed_center Function

### Community 1 - "Force Error Calculation"
Cohesion: 0.83
Nodes (4): calculateForceError_Fixed Function, computeResiduals_Spline Function, Milling Force Simulation with Chip Thickness, Shape-Preserving Piecewise Cubic Interpolation

### Community 2 - "Experimental Data Analysis"
Cohesion: 0.83
Nodes (4): ForceDataAnalysisv11tool15 Script, ForceDataAnalysisv13tool15 Script, Hardstava Tool 15 Experimental Force Data, Tooth-Level Zero-Crossing Sampling Analysis

### Community 3 - "Tool Wear Measurement"
Cohesion: 0.67
Nodes (3): calculate_edge_radii Function, Tool Wear Area Measurement and Visualization, wornArea Main Function

## Knowledge Gaps
- **5 isolated node(s):** `findRefinedAnchor Function`, `run_ransac_fixed_center Function`, `circfit_radius_only Function`, `B-Spline Control Point Force Modeling`, `Tool Wear Area Measurement and Visualization`
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_Spline_Optimizer Script` connect `B-Spline Optimization` to `Force Error Calculation`, `Experimental Data Analysis`?**
  _High betweenness centrality (0.095) - this node is a cross-community bridge._
- **Why does `computeResiduals_Spline Function` connect `Force Error Calculation` to `B-Spline Optimization`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Why does `Hardstava Tool 15 Experimental Force Data` connect `Experimental Data Analysis` to `B-Spline Optimization`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._
- **What connects `findRefinedAnchor Function`, `run_ransac_fixed_center Function`, `circfit_radius_only Function` to the rest of the system?**
  _5 weakly-connected nodes found - possible documentation gaps or missing edges._