import matplotlib.pyplot as plt

# File path
file_path = 'data/platy_beat_test/20240911/ciliate_1fil_1blob_1.00R_0.0010torsion_0.0000tilt_0.3000f_eff_1.4960theta0_true_states.dat'

# Initialize lists to store indices and values
indices = []
values = []

# Read the file
with open(file_path, 'r') as file:
    for line in file:
        parts = line.split()
        index = int(parts[0])
        value = float(parts[2])
        indices.append(index)
        values.append(value)

# Plot the data
plt.plot(indices, values, marker='o')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Value vs Index')
plt.grid(True)

# Save the figure
plt.savefig('output_figure.png')

# Show the plot
plt.show()
