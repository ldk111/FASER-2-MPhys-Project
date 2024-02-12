import os

for i in range(0, 10):
    os.mkdir(str(i))
    for j in range(0, 13):
        try:
            os.mkdir(str(i) + "/" + str(j))
        except FileExistsError:
            print("File Exists")