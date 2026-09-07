# Car Size Classification - Data Preprocessing

## Objective
Prepare the electric-car dataset for multi-class Machine Learning classification.

## Target
`Size` (7 classes): FULL-SIZE, SUBCOMPACT, TWO-SEATER, MID-SIZE, COMPACT, STATION WAGON - SMALL, SUV - STANDARD.

## Preprocessing
- Clean column names
- Remove empty and duplicate rows
- Remove constant/unhelpful columns: TYPE, Unnamed: 5, (g/km), RATING, Model
- Handle missing numerical values with median
- Handle missing categorical values with mode
- One-hot encode categorical features
- Separate target `Size`
- Save processed CSV

## Files
- `cars(2).csv` - original dataset
- `cars_size_classification_preprocessed.csv` - processed dataset
- `preprocessing.py` - preprocessing code

## Run
`pip install pandas`
`python preprocessing.py`

## Note
`TYPE` has only one class (B), so `Size` is used as the multi-class target.
