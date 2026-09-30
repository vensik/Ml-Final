# ML Final Project

A collection of machine learning projects completed as part of the final ML project.

## Projects

| Project                        | Task           |  Best Score | Kaggle Result |
| ------------------------------ | -------------- | ----------: | ------------: |
| [Titanic](./titanic)           | Classification | **0.77511** |   **Top 35%** |
| [House Prices](./house-prices) | Regression     | **0.13032** |   **Top 39%** |
---

## Dataset Setup
### 1. Create data dirs
```bash
mkdir -p titanic/data house-prices/data
```
### 2. Get Datasets
#### Titanic
```bash
(cd titanic/data && kaggle competitions download -c titanic)
(cd titanic/data && unzip titanic.zip && rm titanic.zip)

```
#### House-prices
```bash
(cd house-prices/data && kaggle competitions download -c house-prices-advanced-regression-techniques)
(cd house-prices/data && unzip house-prices-advanced-regression-techniques.zip && rm house-prices-advanced-regression-techniques.zip)
```

## Instructions to get submission
### 1. Clone repo
```bash
git clone https://github.com/vensik/Ml-Final.git
cd Ml-Final
```
### 2. Create and activate venv
```bash
python3 -m venv .venv
source .venv/bin/activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Create and send submit to Kaggle
#### Titanic
```bash
(cd titanic && python main.py && kaggle competitions submit -c titanic -f "$(ls -t results/*submission*.csv | head -1)")
```
#### House-prices
```bash
(cd house-prices && python main.py && kaggle competitions submit -c house-prices-advanced-regression-techniques -f "$(ls -t results/*submission*.csv | head -1)")
```
---

## 🚢 Titanic

**Task:** Binary classification — predict whether a passenger survived the Titanic disaster.

### Feature engineering

- **Initial** - Initials of passenger such as Mr, Miss, etc
- **Age_range** - Age splited in 5 equal folds 
- **Fare_cat** - Same as Age for Fare in 4 folds
- **FamSize** and **Alone** - Combination of Sibling|Spouse and Parent|Children feats. Alone if FamSize = 0

### Result

**Best model on CV:** CatBoost

**Best Kaggle score: 0.77511**

**Leaderboard:** Top 35%

---

## 🏠 House Prices

**Task:** Regression — predict residential property prices based on available property features.

### Feature engineering

- **TotalSF** - Overall house square feet
- **TotalPorchSF** - Overall square feet of open-air zone of the house
- **HouseAge** - Age from built to sold
- **RemodAge** - Age from remodeled to sold
- **TotalBath** - Amount of all house bathrooms
- **AvgRoomArea** - Rude mean square foot of each room

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

