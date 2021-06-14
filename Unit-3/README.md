# Unit-3
---
## Contents
- ### Evaluation
    - #### [Exam U3](https://github.com/OsvaldoHdez/BigData/tree/Unit-3/Unit-3/Exam#exam-u3)


## code

```scala

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
```
