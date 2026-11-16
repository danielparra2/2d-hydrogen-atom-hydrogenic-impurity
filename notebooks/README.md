# Notebook

`atomo_2d_fem.ipynb` is a step-by-step walkthrough of the analytic solution
and the finite-element solver.

It uses the precomputed results in `../results/` and the figures in
`../figures/`; it does not re-run the FEM.

## Run

```bash
pip install -r ../requirements.txt
jupyter lab atomo_2d_fem.ipynb
```

To regenerate the results and figures:

```bash
python ../scripts/validar.py
```
