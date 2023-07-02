# challenges-recsys-challenge-2023


## Preprocess
```python preprocess.py --config submit```

Format all csv files to one parquet file.
- Input
    - train 000000000000.csv ~ 000000000030.csv
    - test 000000000000.csv
- Output
    - train.paruqet
    - test.parquet

## Train & Inference 
```python run.py --config submit --no_wandb --verbose```
- Input 
    - train.parquet
    - test.parquet 
- Output 
    - submission file
    - config.py 
    - feature_importance.png 