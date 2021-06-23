// Import libraries
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors

// 2. Import session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.getOrCreate()

// 3. Load the data stored in LIBSVM format as a DataFrame.
val data  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("/Files/bank-full.csv")

// 4. Process of categorizing the variables type string to numeric. 
val yes = data.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val no = yes.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val newcolumn = no.withColumn("y",'y.cast("Int"))

// 5. Vector is created with the column features 
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")

// 7. Transforms into a new df 
val data2 = assembler.transform(newcolumn)

// 8. Column and label are given a new name 
val featuresLabel = data2.withColumnRenamed("y", "label")

// 9. Select index
val dataIndexed = featuresLabel.select("label","features")
// Index columns

// 10. Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = dataIndexed.randomSplit(Array(0.7, 0.3))

// 11. Logistic Regression
val logisticReg = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFamily("multinomial")
val model = logisticReg.fit(trainingData)
val predictions = model.transform(testData)
val predictionAndLabels = predictions.select($"prediction",$"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)

// 12. Print results
println("Confusion matrix:")
println(metrics.confusionMatrix)
println("Accuracy: " + metrics.accuracy) 
println(s"Test Error: ${(1.0 - metrics.accuracy)}")