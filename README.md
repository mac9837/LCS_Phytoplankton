# LCS_Phytoplankton_Diversity
Code and data for exploring the relationship between repelling Lagrangian Coherent Structures and phytoplankton diversity

To run analysis, weekly velocity field data must be downloaded from https://zenodo.org/records/15730622?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjkxNTgxMWUxLWRjYzMtNDQxYy05NDFkLWE0NTI5OWJmNGRiMSIsImRhdGEiOnt9LCJyYW5kb20iOiIxMTc0MWQ5MDc3YThkM2U4YjMxMDM0ZWJkNDczY2JkYSJ9.g7nzNgqjx-pWjMxBDCe16XsiLYjR1csXHqy76qiQfIZFQiiNrSL4Y8l-iQUr3DdexeJHnP31ZdK1xFh2gSoKKA

Then files must be run in the following order:
1. particle_tracking_parallelised2.py - Compute Lagrangian trajectories of virtual particles
2. ftle_extraction2.py - Extract Finite Time Lyapunov Exponent field from trajectories
3. lcs_visualisation2.py - Visualise Lagrangian Coherent Structures (LCS) from FTLE field
4. lcs_bc_analysis2.py - Test effect of LCS on pairwise Bray-Curtis dissimilarity of neighbouring coastal phytoplankton communities
