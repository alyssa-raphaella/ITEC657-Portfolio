COMP257/ITEC657 Data Science Portfolio 
===

Name: Lim, Alyssa
ID Number: 54799857

Portfolio 1 - Strava and Cheetah Analysis
-------------------------------------------------------------
Strava
    - a social fitness network, that is primarily used to track cycling and running exercises, using GPS data although alternative types are available. (Reference: https://en.wikipedia.org/wiki/Strava)

Golden Cheetah
    - an open-source data analysis tool primarily written in C++ with Qt for cyclists and triathletes with support for training as well.
    - can connect with indoor trainers and cycling equipment such as cycling computers and power meters to import data.
    - can connect to cloud services.
    - can then manipulate and view the data, as well as analyze it.
    (Reference: https://github.com/GoldenCheetah/GoldenCheetah/blob/master/README.md)

Cyclist data used in this portfolio are from Strava and Cheetah, which were merged to be able to compare and analyse the different variables and their relationships using different kinds of graphs.

What were analyzed in this Portfolio?
    - The assymetry of the distribution of variables was analyzed, using Kernel Density Estimation(KDE) plot.
    - The relationships/correlations of different variables were analyzed using scattered plot diagram. 
    - The distribution of different variables per catergory was visualized through box plots.
    - The difference with the performance in different ride categories (Race, Ride, and Workout).
    - The factors that gives the cyclist more "Kudos", using bar graphs.
    
Other DataFrames in this Portfolio:
    - Total Average Speed, TSS, and Distance per Month
    - A function(input: mm, yyyy) was made to list and visualize the distance per day of the different ride categories.