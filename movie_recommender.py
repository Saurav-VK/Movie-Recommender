from pyspark.sql import SparkSession
from pyspark.sql import functions  as func 
from pyspark.sql.types import StructType , StructField , IntegerType , StringType , LongType
import sys

def computePearsonCorrelationSimilarity(spark , data):
    r1_mean = data.agg(func.mean(func.col("ratings1")).alias("mean1")).collect()[0]["mean1"]
    r2_mean = data.agg(func.mean(func.col("ratings2")).alias("mean2")).collect()[0]["mean2"]
    cols = data \
        .withColumn("num1" , (func.col("ratings1") - r1_mean)) \
        .withColumn("num2" , (func.col("ratings2") - r2_mean)) \
        .withColumn("tot" , func.col("ratings1") * func.col("ratings2"))
              
    calculateSimilarity = cols.groupBy("movie1" , "movie2").agg( \
                                                                func.sum(func.col("num1") * func.col("num2")).alias("numerator"), \
                                                                 (func.sqrt((func.sum(func.col("num1") * func.col("num1"))) * (func.sum(func.col("num2") * func.col("num2"))))).alias("denominator"), \
                                                                    func.count(func.col("tot")).alias("numPairs"))
    result = calculateSimilarity.withColumn("score" , \
                                            func.when(func.col("denominator") != 0 , func.col("numerator") / func.col("denominator")).otherwise(0)).select("movie1" , "movie2" , "score" , "numPairs")
    return result

def GetName(nameData , mid):
    result = nameData.filter(func.col("mid") == mid).select("title").collect()[0]
    return result[0]

spark = SparkSession.builder.appName("movie-recommender").getOrCreate()

dataSchema  = StructType([ \
                         StructField("uid" , IntegerType() , True), \
                         StructField("mid" , IntegerType() , True), \
                         StructField("ratings" , IntegerType() , True), \
                         StructField("ts" , LongType() , True) ]) 
nameSchema = StructType([ \
                         StructField("mid" , IntegerType() , True), \
                         StructField("title" , StringType() , True)])

movieData = spark.read.option("sep" , "\t") \
            .schema(dataSchema).csv("C:\\SparkCourse\\ml-100k\\u.data")

nameData = spark.read.option("sep" , "|").option("charset" , "ISO-8859-1").schema(nameSchema) \
           .csv("C:\\SparkCourse\\ml-100k\\u.item")

nameData.show()

ratings_data = movieData.select("uid" , "mid" , "ratings")

moviePairs = ratings_data.alias("ratings1").join(ratings_data.alias("ratings2") , \
             (func.col("ratings1.uid") == func.col("ratings2.uid")) \
             & (func.col("ratings1.mid") < func.col("ratings2.mid"))) \
             .select(func.col("ratings1.mid").alias("movie1"), \
              func.col("ratings2.mid").alias("movie2"), \
              func.col("ratings1.ratings").alias("ratings1"), \
              func.col("ratings2.ratings").alias("ratings2"))

moviePairs.show()

similarities_df = computePearsonCorrelationSimilarity(spark , moviePairs).cache()
similarities_df.show()

if len(sys.argv) > 1:
    threshold = 0.80
    min_ratings = 20.0
    
    movieId = int(sys.argv[1])
    
    filtered_data = similarities_df.filter( \
                    ((func.col("movie1") == movieId) | (func.col("movie2") == movieId)) \
                    & (func.col("score") > threshold) & (func.col("numPairs") > min_ratings))
    filtered_data.show()
    
    similarmovies = filtered_data.sort(func.col("score").desc()).take(10)
    
    
    print("THE TOP 10 MOVIES SIMILAR TO " + GetName(nameData , movieId) + " ARE:")
    
    for movie in similarmovies:
        similar_ID = movie.movie1
        if similar_ID == movieId:
            similar_ID = movie.movie2
            
        print(GetName(nameData , similar_ID) + "SCORE: " + str(movie.score) + " NUMBER OF USER RATINGS: " + str(movie.numPairs) +"\n")

        
    