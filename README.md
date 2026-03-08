# Phase-diagram-Visualizing-the-criterion-in-parameter-space
This code simulates social contagion dynamics on hypergraphs to validate the η-criterion for explosive transitions.
The code simulates social contagion on hypergraphs to test the criterion that a transition is explosive when the ratio η/ηₑ exceeds one. The workflow proceeds as follows:

Parameter setup – The user chooses a network size, the resolution of the structural overlap T and heterogeneity β grids, a list of nonlinearity ratios λ₂/λ₁, and the number of independent realisations per parameter point. Physical constants that control the influence of T and β on the critical threshold are also defined.

Hypergraph generation – For each combination of T and β, a random graph is built by adding edges with a fixed probability and then adding triangles until the desired overlap T is approximately reached. Each node is assigned a weight that depends on β and its degree rank, following the prescribed heterogeneity model.

SIS dynamics – The mean‑field equations are integrated forward in time using an explicit Euler scheme until the system settles into a low‑activity steady state. Both pairwise (linear) and triangular (nonlinear) infection terms are taken into account.

Computation of η and ηₑ – At the steady state, the linearised matrix is formed and its spectral radius (the largest real part of its eigenvalues) is estimated by a combination of Gershgorin’s circle theorem and power iteration with restarts. The nonlinear drive is obtained from the triangle contributions and normalised by the norm of the linearised matrix. The theoretical critical value ηₑ is calculated from the network structure, the node weights, and the input β, using the empirical constants. Finally the ratio η/ηₑ is clipped to a predefined range.

Explosive classification – A single realisation is flagged as explosive if η/ηₑ > 1; hysteresis is computed but not used for the decision, in line with the theoretical prediction.

Statistical aggregation – Results are grouped by λ₂/λ₁, T and β. For each group the mean value of η/ηₑ, the fraction of explosive runs, and the 95 % confidence intervals are calculated using the t‑distribution.

Visualisation –
Phase diagrams (Figure 1) are produced by interpolating the mean η/ηₑ onto a fine grid and displaying it with a red‑blue diverging colour map (red for η/ηₑ > 1, blue for < 1). The critical contour η/ηₑ = 1 is overlaid as a red dashed line and individual simulation points are shown as coloured dots.
Statistical plots (Figure 2) show the evolution of the explosive probability with λ₂/λ₁, the distribution of η/ηₑ for each nonlinearity, and the separate effects of T and β, complete with error bars.

The entire simulation is parallelised over multiple CPU cores and offers a test mode for rapid checks as well as a full production mode for final results.

The workflow described above directly corresponds to the implementation in eta_eta_c_SIS_hotmap.py.  Generated: Fig. 1 in the main text, and Fig. H1 in Appendix H.
