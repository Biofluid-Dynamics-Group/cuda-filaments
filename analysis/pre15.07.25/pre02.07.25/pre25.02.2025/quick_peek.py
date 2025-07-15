import matplotlib.pyplot as plt

with open('data/yestiltgap_g10.0/20241202/ciliate_308fil_5000blob_4.00R_0.0010torsion_0.2182tilt_0.3000f_eff_1.4960theta0_blob_references.dat', 'r') as file:
    data = file.read().split()

# Interpret values as 3D positions
positions = [(float(data[i]), float(data[i+1]), float(data[i+2])) for i in range(0, len(data), 3)]

# Extract x, y, and z coordinates
x = [pos[0] for pos in positions]
y = [pos[1] for pos in positions]
z = [pos[2] for pos in positions]

# Plot the positions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Positions in 3D Space')

print("hi")

# Show the plot
plt.savefig('peek.png')