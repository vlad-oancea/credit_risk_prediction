# Credit risk

## Instructions
Once you have cloned this repo, you can run the following commands at the root of the repo:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install numpy matplotlib pandas seaborn xgboost imbalanced-learn scikit-learn tensorflow requests openpyxl
python3 code/submission.py
```
This generates three folders: 
- a `data` folder with the dataset
- a `figures` folder with the plots
- a `tables` folder with .xlsx files to display the results

**Disclaimer Concerning the Monte Carlo simulation plot :** When reorganizing the final submission .py file from the notebook we worked on, the seed we used was lost, which might lead to different results. This does not compromise the validity of the resoning and the interpretation in our paper.