# Skin-Cancer-Diagnosis-System-From-Images-With-Machine-Learning
We worked on machine learning methods to detect skin cancer. 
In this project, the ABCD method was used  on the skin cancer detection as by dermatologists. 
For Machine Learninng KNN algorithm was used the nearest neighbor algorithm to determine whether the disease is sick or healthy.
For Features the number of points belonging to the irregular lengths from the center to the boundary for the data to be given,
the number of high jumping points in the border found by the derivative, the standard deviation of the color,
features such as similarity, similarity ratio, number of contours were used.

Running the project:

1) Images of lesion areas send to the "photo" folder.
2) By running the Hair_Filtering.py file, the images are processed and submitted to the "foto2" folder.
3) The main.py file is run and the attributes of the photos in the "foto2" folder are written to the les.csv file.
4) Run the knn.py file. With the data from the .csv file, the knn algorithm is estimated.
