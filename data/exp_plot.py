import matplotlib.pyplot as plt


file_paths = ['pn_v16_t300.txt', 'pt_n1000_v16.txt', 'pv_n1000_t300.txt', 'temp-time_shake.txt', 'temp-time_vfix.txt']
titles = ['P-N', 'P-T', 'P-1/V', 'Temperature with volume change', 'Temperature with volume fixed']

# Read the data files
def read_data(file_path, use_third_value=False):
    pressures = []
    values = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                parts = line.split(',')
                pressure = float(parts[0])
                if use_third_value:
                    value = float(parts[2])  # only use the third value when reading the third data file
                else:
                    value = float(parts[1])  # default for using the second value
                pressures.append(pressure)
                values.append(value)
    return pressures, values

# Subplot
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(14, 10))
ax6.axis('off') 

# Setting the title and name of x-y axis
pressures, values = read_data(file_paths[0])
ax1.scatter(pressures, values, color='blue')
ax1.set_title(titles[0])
ax1.set_xlabel('Pressure (P)')
ax1.set_ylabel('Number of Particles (N)')
ax1.grid(True)

pressures, values = read_data(file_paths[1])
ax2.scatter(pressures, values, color='green')
ax2.set_title(titles[1])
ax2.set_xlabel('Pressure (P)')
ax2.set_ylabel('Temperature (T)')
ax2.grid(True)

pressures, values = read_data(file_paths[2], use_third_value=True)
ax3.scatter(pressures, values, color='red')
ax3.set_title(titles[2])
ax3.set_xlabel('Pressure (P)')
ax3.set_ylabel('Reciprocal of volume (1/V)')
ax3.grid(True)

pressures, values = read_data(file_paths[3])
ax4.scatter(pressures, values, color='purple')
ax4.set_title(titles[3])
ax4.set_xlabel('Temperature (T)')
ax4.set_ylabel('Time (sec)')
ax4.grid(True)

pressures, values = read_data(file_paths[4])
ax5.scatter(pressures, values, color='orange')
ax5.set_title(titles[4])
ax5.set_xlabel('Temperature (T)')
ax5.set_ylabel('Time (sec)')
ax5.grid(True)

plt.tight_layout()
plt.show()