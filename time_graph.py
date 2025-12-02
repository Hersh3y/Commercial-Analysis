from matplotlib import pyplot as plot

# Data for individual tasks
task_categories = ['Audio\nExtraction', 'Transcription', 'Celebrity\nQuestions', 'Demographic\nQuestions', 'Brand\nQuestions']
task_times = [0.05, 2.17, 41, 31, 67]
task_labels = ['0.05s', '2.17s', '41s', '31s', '67s']

# Create bar graph
fig, ax = plot.subplots(figsize=(10, 6))
bars = ax.bar(task_categories, task_times, color='steelblue', edgecolor='black')
ax.set_xlabel('Task', fontsize=12, labelpad=10)
ax.set_ylabel('Time (seconds)', fontsize=12)
ax.set_title('Processing Time for each task', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, label in zip(bars, task_labels):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2, label,
            ha='center', va='bottom', fontweight='bold')

plot.tight_layout()
plot.subplots_adjust(bottom=0.1)
plot.show()
