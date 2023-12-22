from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import FileLink
import csv

# Example evaluation metrics data
models = ['Logistic regression', 'Decision Tree', 'Naive Bayes', 'Random Forest', 'XGBoost', 'AdaBoost(DT)', 'Stacking(DT+XGB+RF)', 'Voting(DT+XGB+RF)','Bagging(DT)']
accuracy = ['0.7825036985466887','0.8944500043512315','0.7368157688625881','0.8288225567835698','0.9490905926377164','0.8967887912279175','0.9244843790792794','0.9284548777303977','0.920100513445305']
precision = ['0.7645464917284654','0.905958218579969','0.7478821278921479','0.7954030177468533','0.98764469643279','0.8978621291448516','0.9539342050160532','0.950559381363106','0.9614750376407045']
recall = ['0.8164433034548777','0.880275868070664','0.7144939517883561','0.8853885649638847','0.9095596553824732','0.8954399094943869','0.892045949003568','0.9039248107214342','0.8752719519624054']
f1_score = ['0.7896431276828549','0.8929324137931034','0.7308068894921893','0.837986986244955','0.9469952658164769','0.8966493834691298','0.9219526454253142','0.9266557380705452','0.9163506326374891']

# Combine the data into a list of lists
data = [models, accuracy, precision, recall, f1_score]

# Transpose the data so that each model's metrics are in a row
data = list(map(list, zip(*data)))

# Print the table using tabulate
headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
print(tabulate(data, headers=headers, tablefmt='orgtbl'))
# Write table to CSV file
with open('heartoutput.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
    writer.writerows(data)

# Make table downloadable
display(FileLink('heartoutput.csv'))

# Convert the metric scores to float and round off to two decimal places
accuracy = [round(float(score), 16) for score in accuracy]
precision = [round(float(score), 2) for score in precision]
recall = [round(float(score), 2) for score in recall]
f1_score = [round(float(score), 10) for score in f1_score]


# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot the accuracy, precision, recall, and f1-score as bars in the first subplot
bar_width = 0.2
x = np.arange(len(models))

ax1.bar(x - 1.5*bar_width, accuracy, bar_width, label='Accuracy')
ax1.bar(x - 0.5*bar_width, precision, bar_width, label='Precision')
ax1.bar(x + 0.5*bar_width, recall, bar_width, label='Recall')
ax1.bar(x + 1.5*bar_width, f1_score, bar_width, label='F1-Score')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.set_xlabel('Models')
ax1.set_ylabel('Metric Scores')
ax1.set_title('Evaluation Metrics for Different Models')
ax1.legend()

# Plot the accuracy, precision, recall, and f1-score as lines in the second subplot
ax2.plot(models, accuracy, label='Accuracy')
ax2.plot(models, precision, label='Precision')
ax2.plot(models, recall, label='Recall')
ax2.plot(models, f1_score, label='F1-Score')
ax2.set_xticklabels(models, rotation=45, ha='right')
ax2.set_xlabel('Models')
ax2.set_ylabel('Metric Scores')
ax2.set_title('Evaluation Metrics for Different Models')
ax2.legend()

# Adjust the layout and display the figure
fig.tight_layout()
plt.show()
