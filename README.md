 # Self-Adapted Entity-Centric Data Augmentation for Discontinuous Named Entity Recognition

## 1. Environments

```
- python (3.8.12)
```

## 2. Dependencies

```
- numpy (1.23.0)
- torch (2.0.0+cu117)
- transformers (4.33.0)
- pandas (1.3.3)
- scikit-learn (1.3.0)
```

## 3. Dataset

- [CADEC](https://pubmed.ncbi.nlm.nih.gov/25817970/)
- [ShARe13](https://clefehealth.imag.fr/?page_id=441)
- [ShARe14](https://sites.google.com/site/clefehealth2014/)

## 4. Preparation

- Download dataset
- Process them to fit the same format as the example in `data/`
- Put the processed data into the directory `data/`

## 5. Training

```bash
>> python main.py --config ./config/cadec.json
>> python main.py --config ./config/share13_fix_length.json
>> python main.py --config ./config/share14_fix_length.json
```

"# SEDA" 
