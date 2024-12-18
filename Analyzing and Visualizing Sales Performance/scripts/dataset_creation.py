import pandas as pd
import numpy as np
from faker import Faker
import os

# Initialize Faker and random seed
faker = Faker()
np.random.seed(42)

# Parameters for dataset generation
n_rows = 10000
products = ["Laptop", "Smartphone", "Tablet", "Desk", "Chair", "Monitor", "Headphones", "Keyboard", "Mouse", "Printer"]
categories = {
    "Electronics": ["Laptop", "Smartphone", "Tablet", "Monitor", "Headphones", "Keyboard", "Mouse", "Printer"],
    "Furniture": ["Desk", "Chair"],
}
regions = ["North", "South", "East", "West"]
priorities = ["Low", "Medium", "High", "Critical"]

# Generate data
data = {
    "OrderID": np.arange(1, n_rows + 1),
    "OrderDate": [faker.date_between(start_date="-1y", end_date="today") for _ in range(n_rows)],
    "Product": np.random.choice(products, size=n_rows),
    "Revenue": np.random.randint(50, 5000, size=n_rows),
    "Cost": np.random.randint(30, 4000, size=n_rows),
    "Quantity": np.random.randint(1, 20, size=n_rows),
    "CustomerID": np.random.randint(1000, 5000, size=n_rows),
    "Region": np.random.choice(regions, size=n_rows),
    "OrderPriority": np.random.choice(priorities, size=n_rows),
    "Discount": np.random.randint(0, 30, size=n_rows),  # Discount in percentage
}

# Map products to categories
data["Category"] = [next(k for k, v in categories.items() if product in v) for product in data["Product"]]

# Calculate Profit
data["Profit"] = data["Revenue"] - data["Cost"]

# Create a DataFrame
sales_data = pd.DataFrame(data)

# File path
directory = r"E:\MatPlotLib-SeaBorn Project\Analyzing and Visualizing Sales Performance\data"
file_path = os.path.join(directory, "sales_data.csv")

# Ensure the directory exists
os.makedirs(directory, exist_ok=True)

# Save the dataset
sales_data.to_csv(file_path, index=False)

print(f"Dataset with {n_rows} rows saved to {file_path}")
