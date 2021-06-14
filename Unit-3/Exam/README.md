# Exam U3
### 1. Import a simple Spark session.
#### Code
```scala
// 1. Import a simple Spark session.
import org.apache.spark.sql.SparkSession
```

### 2. Use the lines of code to minimize errors.
#### Code
```scala
// 2. Use the lines of code to minimize errors
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
```

### 3. Create an instance of the Spark session.
#### Code
```scala
// 3. Create an instance of the Spark session
val spark = SparkSession.builder().getOrCreate()
```

### 4. Import the Kmeans library for the grouping algorithm.
#### Code
```scala
// 4. Import the Kmeans library for the grouping algorithm.
import org.apache.spark.ml.clustering.KMeans
```

### 5. Load the Wholesale Customers Data dataset.
#### Code
```scala
// 5. Load the Wholesale Customers Data dataset
val dataset = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Wholesale customers data.csv")
dataset.show
```
#### Results 
<html><div align="center"><img src="https://i.ibb.co/nr8jDHb/imagen.png"></div></html>

### 6. We select the following columns: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen and call this set feature_data.
#### Code
```scala
// 6. We select the following columns: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen and call this set feature_data
val feature_data  = dataset.select("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")
feature_data.show
```
#### Results 
<html><div align="center"><img src="https://i.ibb.co/VDvPK5j/imagen.png"></div></html>

### 7. Import Vectorassembler and Vector.
#### Code
```scala
// 7. Import Vectorassembler and Vector
import org.apache.spark.ml.feature.VectorAssembler
```

### 8. Create a new Vector Assembler object for the feature columns as an input set, remembering that there are no labels.
#### Code
```scala
// 8. Create a new Vector Assembler object for the feature columns as an input set, remembering that there are no labels
val assembler = new VectorAssembler().setInputCols(Array("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")).setOutputCol("features")
```

### 9. Use the assembler object to transform feature_data.
#### Code
```scala
// 9. Use the assembler object to transform feature_data
val features = assembler.transform(feature_data)
features.show
```
#### Results 
<html><div align="center"><img src="https://i.ibb.co/Yp6Q5d3/imagen.png"></div></html>

### 10. Create a Kmeans model with K = 3.
#### Code
```scala
// 10. Create a Kmeans model with K = 3
val kmeans = new KMeans().setK(3).setSeed(1L)
val model = kmeans.fit(features)
```

### 11. Evaluate the groups using Within Set Sum of Squared Errors WSSSE and print the centroids.
#### Code
```scala
// 11. Evaluate the groups using Within Set Sum of Squared Errors WSSSE and print the centroids.
val WSSSE = model.computeCost(features)
println(s"Within Set Sum of Squared Errors = $WSSSE")

    // Print cluster centers
    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)
```
#### Results 
<html><div align="center"><img src="https://i.ibb.co/vmJkwz1/imagen.png"></div></html>