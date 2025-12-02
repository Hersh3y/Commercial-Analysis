from matplotlib import pyplot as plot

# Data
categories = ['Celebrity', 'Human Count', 'Gender', 'Brand Name', 'Brand/Logo\nAppearance', 'Ad Message\nand Theme']
accuracies = [81.25, 43.75, 62.5, 100, 50, 100]

# Create bar graph
plot.figure(figsize=(10, 6))
plot.bar(categories, accuracies, color='red', edgecolor='black')
plot.xlabel('Category', fontsize=12)
plot.ylabel('Accuracy (%)', fontsize=12)
plot.title('Model Accuracies for each category', fontsize=14, fontweight='bold')
plot.ylim(0, 110)
plot.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(accuracies):
    plot.text(i, v + 2, f'{v}%', ha='center', fontweight='bold')

plot.tight_layout()
plot.show()

