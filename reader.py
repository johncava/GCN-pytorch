def read(file_name):
    with open(file_name) as file:
        matrix = []
        for line in file:
            array = []
            line = line.split()
            for vertex in line:
                array.append(float(vertex))
            matrix.append(array)
    return matrix

def read_data(input, label):
    return read(input), read(label)