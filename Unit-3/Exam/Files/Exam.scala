// 1. Import a simple Spark session.
import org.apache.spark.sql.SparkSession
 
// 2. Use the lines of code to minimize errors
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
 
// 3. Create an instance of the Spark session
val spark = SparkSession.builder().getOrCreate()
 
// 4. Import the Kmeans library for the grouping algorithm.
import org.apache.spark.ml.clustering.KMeans
 
// 5. Load the Wholesale Customers Data dataset
val dataset = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Wholesale customers data.csv")
dataset.show
 
// 6. We select the following columns: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen and call this set feature_data
val feature_data  = dataset.select("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")
feature_data.show

// 7. Import Vectorassembler and Vector
import org.apache.spark.ml.feature.VectorAssembler

// 8. Create a new Vector Assembler object for the feature columns as an input set, remembering that there are no labels
val assembler = new VectorAssembler().setInputCols(Array("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")).setOutputCol("features")

// 9. Use the assembler object to transform feature_data
val features = assembler.transform(feature_data)
features.show

// 10. Create a Kmeans model with K = 3
val kmeans = new KMeans().setK(3).setSeed(1L)
val model = kmeans.fit(features)

// 11. Evaluate the groups using Within Set Sum of Squared Errors WSSSE and print the centroids.
val WSSSE = model.computeCost(features)
println(s"Within Set Sum of Squared Errors = $WSSSE")

Print cluster centers
println("Cluster Centers: ")
model.clusterCenters.foreach(println)