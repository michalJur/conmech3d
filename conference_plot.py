import matplotlib.pyplot as plt

categories = ['Baseline\n(skinning)', 'GNN']

values = [0.01832, 0.01772]
title ='slide'

values = [0.01087, 0.01041]
title ='pingpong'

values = [0.0043172, 0.0042519]
title ='chute'


scalar = 1000
values = [v * scalar for v in values]
shift = values[0]/40. # 0.0005


# Create the bar graph
plt.figure(figsize=(4, 6))
bars = plt.bar(categories, values, color=['#1f77b4', '#ff7f0e'])  # Explicitly set blue and orange
plt.xticks(fontsize=18)  # Increase x-axis (category) label font size
plt.yticks(fontsize=12) 

# Customize the plot
plt.title(f'Simulation: {title}', fontsize=16)  # Main title
# plt.xlabel('Categories', fontsize=15)      # X-axis label
plt.ylabel(f'MSE X {scalar}', fontsize=16)          # Y-axis label

# # Add value labels on top of bars (optional)
# for i, bar in enumerate(bars):
#     plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), # + 0.5,
#              str(values[i]), ha='center', va='bottom')

# Add vertical percentage difference arrow between bars
if len(values) == 2:
    # Calculate percentage difference
    diff_percent = abs((values[1] - values[0]) / values[0] * 100)
    print(f"Percentage difference: {diff_percent:.1f}%")
    
    # Get bar positions and heights
    bar1_x = bars[0].get_x() + bars[0].get_width() / 2
    bar2_x = bars[1].get_x() + bars[1].get_width() / 2
    bar1_height = bars[0].get_height()
    bar2_height = bars[1].get_height()
    
    # Position vertical arrow between the bars
    arrow_x = (bar1_x + bar2_x) / 2
    
    # Draw vertical arrow from top of one bar to top of the other
    # plt.annotate('', xy=(arrow_x, bar2_height), xytext=(arrow_x, bar1_height),
    #             arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    
    # Add percentage text to the left of the arrow
    mid_y = (bar1_height + bar2_height) / 2
    plt.text(arrow_x, mid_y+shift, f'{diff_percent:.1f}% diff.', 
             ha='left', va='center', fontsize=15)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig(f'bar_graph_{title}.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()
plt.close()

print("Bar graph saved'")