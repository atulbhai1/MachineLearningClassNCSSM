"""
Created By: Atul Nitin
Assignment Number: P0
Created Date: Aug. 24th, 2026
Last Modified: Aug. 24th, 2026
Requirements:
- pandas
- numpy
- matplotlib
"""
#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

#Drag data into pandas because csv reader keeps the top columns, and I don't like that
pandas_raw = pd.read_csv("iris_data.csv")

#I really like pandas, but here is my reluctantly created numpy code 😢

#Turn into Array for numpy stuff
iris_data = np.array(pandas_raw)

#Make empty lists to add on the individual rows to in the for loop
setosa = []
virginica = []
versicolor = []
for row in iris_data:
    if row[4] == "setosa":
        setosa.append(list(row))
    elif row[4] == "virginica":
        virginica.append(list(row))
    elif row[4] == "versicolor":
        versicolor.append(list(row))

#Turn lists into arrays for usability
virginica = np.array(virginica)
setosa = np.array(setosa)
versicolor = np.array(versicolor)

#Plot graph for part 2(max 10 tick marks!!!)
plt.plot(virginica[:, 0], virginica[:, 2], ".", color="red", label="virginica")
plt.plot(setosa[:, 0], setosa[:, 2], "x", color="blue", label="setosa")
plt.plot(versicolor[:, 0], versicolor[:, 2], "o", color="green", label="versicolor")

plt.xlabel("Sepal Width(cm)")
plt.ylabel("Petal Width(cm)")

plt.title("Iris Setosa Sepal to Petal Widths - Part 2")
plt.legend(loc='lower right')

plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=10))


plt.show()

#Column Names at indexes corresponding to i and j
column_names = ["Sepal Length(cm)", "Sepal Width(cm)", "Petal Length(cm)", "Petal Width(cm)"]
combos = []#By keeping track of what has been done, we can avoid graphing duplicates like sepal length x sepal width and then sepal width x sepal length
for i in range(4):#For each of the four attributes(x axis)
    for j in range(4):#For each of the four attributes(y axis)
        if i!=j:#Enusre you aren't graphing something against itself
            #The below algorithm checks if this combo has been seen before. If so, tracker becomes false
            tracker = True
            for combo in combos:
                if combo == {i, j}:
                    tracker = False
            #If this is a unique combo, it will plot the graph for all three species under different colors and dot types(max 10 tick marks!!!)
            if tracker:

                combos.append({i, j})
                for (flower, name, color, marker) in [(setosa, "Setosa", "blue", "x"), (virginica, "Virginica", "red", "."), (versicolor, "Versicolor", "green", "o")]:
                    plt.plot(flower[:, i], flower[:, j], marker, color=color, label=name)

                title = column_names[i] + " by " + column_names[j] + " For All Flowers - Part 3"#Makes a title

                plt.title(title)
                plt.xlabel(column_names[i])
                plt.ylabel(column_names[j])
                plt.legend(loc='upper left')
                plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))
                plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=10))
                plt.show()#Generates Graph

#In total, there should be 7 graphs: one for part 2 and then six distinct ones to show all the data

#NOTE: FOR SOME REASON THE POINTS ARE ALL MESSED UP AXIS WISE(PRBLY BECAUSE DIFF COLORS ARE IN THE SAME GRAPH)
#NOTE: SCATTER COULDN'T FIX IT AND SORTING BOTH AXES CAN UNCOUPLE CORRESPONDING VALUES, RUINING THE DATA. FIXING JUST 1 AXIS IS SOMEWHAT USELESS IN MY OPINION SO I LEFT IT AS IS



