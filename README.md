# Movie-Recommender
This project implements a movie recommender system using Apache Spark and PySpark, focusing on calculating movie similarity based on user ratings. The system leverages Pearson correlation to measure the similarity between movie pairs, providing recommendations based on a specified movie.
The dataset used is the ml-100k dataset that can be downloaded from grouplens.org -> datasets.
The details about the dataset necessary for the program such as delimiters and contents are availabale in the readme.md file of the dataset folder.
Pre-requisites: Apache pyspark must be installed.
To execute the program: run the following command: spark-submit movie_recommender.py [movie Id of the desired movie]
