Testing EL-GNN on U
```bash
python ray_hpo.py
```

To test QL-GNN on other datasets, please set `line 185: 'degree': tune.choice([1, 3, 5, 7, 9])` to `'degree': tune.choice([np.inf])`.