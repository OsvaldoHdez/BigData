//Import a simple Spark session.
 
import org.apache.spark.sql.SparkSession
 
//Use the lines of code to minimize errors
 
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
 
//We create an instance of the Spark session
val spark = SparkSession.builder().getOrCreate()
 
//We import the Kmeans library for the grouping algorithm.
import org.apache.spark.ml.clustering.KMeans
 
//We load the Wholesale Customers Data dataset

val dataset = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Wholesale customers data.csv")
dataset.show
 
//We select the following columns: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen and call this set feature_data
val  feature_data  = dataset.select("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")
feature_data.show
