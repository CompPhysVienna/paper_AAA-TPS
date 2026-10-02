# Renumber atoms in a .gro file and save it with new atom indices

def renumber_gro(input_file, output_file):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    # Open the output file to write the renumbered data
    with open(output_file, 'w') as outfile:
        # Copy the first two lines (box size and total atom count)
        outfile.writelines(lines[:2])

        # Initialize the new atom index
        new_index = 0
        second_index = 1

        number = lines[2].split()[0]
        number = ''.join([char for char in number if char.isdigit()])
        
        # Loop through the atom lines (skip the first two lines)
        for line in lines[2:]:
            if line.strip():  # Ignore empty lines
                parts = line.split()
                if len(parts) > 3:  # If the line contains atom data
                    # Replace the old atom index with the new one#

                    print(parts)
                    current_number = parts[0]
                    current_number = ''.join([char for char in current_number if char.isdigit()])
                    current_type = parts[0]
                    current_type = ''.join([char for char in current_type if char.isalpha()])
                    print(current_number)

                    if number != current_number:
                        new_index += 1

                    number = current_number

                    # Loop through the data and print each line using the format string
                    outfile.write("{:5d}{:<5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}\n".format(
                        new_index, current_type, f"{parts[1]}", second_index, float(parts[3]), float(parts[4]), float(parts[5])
                    ))
                    second_index += 1
                    
                else:
                    # If the line doesn't contain atom data (box size, etc.), just write it as it is
                    outfile.write(line)

# Specify input and output file names
input_file = 'cage4151062.gro'  # Replace with the path to your input .gro file
output_file = 'reindexed_cage4151062.gro'  # Desired output file name

# Call the function to renumber the atoms
renumber_gro(input_file, output_file)

print(f"Renumbered .gro file saved as {output_file}")
