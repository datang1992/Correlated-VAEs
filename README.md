# Correlate-Variational-Auto-Encoders
Code for my ICML 2019 paper [Correlated Variational Auto-Encoders](https://arxiv.org/abs/1905.05335)
## Files
- **cvae_ind.py**: Code for the algorithm CVAE<sub>ind</sub> on general graphs (Section 4.2.3).
- **cvae_corr.py**: Code for the algorithm CVAE<sub>corr</sub> on general graphs (Section 4.2.3).
- **process_tree_data.py**: Code for constructing the synthetic dataset for the spectral clustering experiment (Section 4.2.2).
- **process_Epinions_data.py**: Code for preprocessing the Epinions dataset for the general graph link prediction experiment (Section 4.2.3). To use this code, construct an NumPy npz file that contains two arrays with values from the two datasets (ratings_data and trust_data) on the [Epinions dataset website](http://www.trustlet.org/downloaded_epinions.html) [1] and run this code with the argument *input_data_file_name* being set as the npz file directory.
---
### References

[1] Trust-aware recommender systems. P Massa, P Avesani. Proceedings of the 2007 ACM conference on Recommender systems, 17-24
