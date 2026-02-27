import matplotlib.pyplot as plt
from data_loader import load_data

df = load_data("../data/sales_data.csv")

plt.figure()
plt.plot( df['date'], df['sales'] )
plt.title("Sales over Time")
plt.xticks (rotation = 45)
plt.tight_layout()
plt.show()