# ML Final Project

A collection of machine learning projects completed as part of the final ML project.

## Projects

| Project                        | Task           |  Best Score | Kaggle Result |
| ------------------------------ | -------------- | ----------: | ------------: |
| [Titanic](./titanic)           | Classification | **0.77511** |   **Top 35%** |
| [House Prices](./house-prices) | Regression     | **0.13032** |   **Top 39%** |

## Instructions to get submission
    1. cd titanic or house-prices
    2. source .venv/bin/activate
    3. pip install requirements.txt
    4. Check config.py if cv -> ON is False
    5. titanic: 
    python main.py && kaggle competitions submit -c titanic -f "$(ls -t results/*submission*.csv | head -1)
       house-prices: python main.py && kaggle competitions submit -c house-prices-advanced-regression-techniques -f "$(ls -t results/*submission*.csv | head -1)"

---

## 🚢 Titanic

**Task:** Binary classification — predict whether a passenger survived the Titanic disaster.

### Feature engineering

**Initial** - Initials of passenger such as Mr, Miss, etc
**Age_range** - Age splited in 5 equal folds 
**Fare_cat** - Same as Age for Fare in 4 folds
**FamSize** and **Alone** - Combination of Sibling|Spouse and Parent|Children feats. Alone if FamSize = 0

### Result

**Best model on CV:** CatBoost

**Best Kaggle score: 0.77511**

**Leaderboard:** Top 35%

---

## 🏠 House Prices

**Task:** Regression — predict residential property prices based on available property features.

### Feature engineering

**TotalSF** - Overall house square feet
**TotalPorchSF** - Overall square feet of open-air zone of the house
**HouseAge** - Age from built to sold
**RemodAge** - Age from remodeled to sold
**TotalBath** - Amount of all house bathrooms
**AvgRoomArea** - Rude mean square foot of each room

### Result

**Best model on CV:** XGBoost

**Best Kaggle score: 0.13032**

**Leaderboard:** Top 39%

---

## 🛠️ Dependencies

* Python
* NumPy
* Pandas
* Scikit-learn
* CatBoost
* LightGBM
* XGBoost
* PyTorch
* Matplotlib
* Seaborn
* OmegaConf

