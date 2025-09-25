import numpy as np
import pandas as pd
from pathlib import Path

def solve_homework1() -> None:

  script_dir = Path(__file__).parent
  df = pd.read_csv(script_dir / 'car_fuel_efficiency.csv')

  # question 1
  print(f"[Question1] pandas version: {pd.__version__}")

  # question 2
  print(f'[Question2] number of records: {len(df)}')

  # question 3
  print(f"[Question3] number fuel types: {df['fuel_type'].nunique()}")

  # question 4
  print(f'[Question4] number of columns with missing values: {sum(df.isnull().sum() > 0)}')

  # question 5
  max_fuel_eff = df.groupby('origin')['fuel_efficiency_mpg'].max()['Asia'].item()
  print(f'[Question5] maximum fuel efficiency of cars from Asia: {max_fuel_eff}')

  # question 6
  print(f"[Question1] median of horsepower before: {df['horsepower'].median()}")
  mfv = df['horsepower'].mode()[0].item()
  df['horsepower'] = df['horsepower'].fillna(mfv)
  print(f"[Question6] median of horsepower after: {df['horsepower'].median()}")

  # question 7
  X = df[df['origin'] == 'Asia'][['vehicle_weight', 'model_year']].iloc[:7].to_numpy()
  xTx = np.matmul(X.T, X)
  y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
  w = np.matmul(np.matmul(np.linalg.inv(xTx), X.T), y)
  print(f"[Question7] final result: {np.sum(w)}")
