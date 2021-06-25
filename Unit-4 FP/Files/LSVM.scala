// 1. Import the "LinearSVC" library, this binary classifier optimizes the hinge loss using the OWLQN optimizer. 
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics

// 2. Import session.
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.getOrCreate()

// 3. Load the training data. 
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
val Array(training, test) = dataIndexed.randomSplit(Array(0.7, 0.3))

// 11. Set the maximum number of iterations and the regularization parameter .
val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)

// 12. Make a fit to adjust the model.
val supportVM = new LinearSVC().setMaxIter(10).setRegParam(0.1)
val model = supportVM.fit(training)
val predictions = model.transform(test)
val predictionAndLabels = predictions.select($"prediction",$"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)

// 13. Print LSVC
println("Confusion matrix:")
println(metrics.confusionMatrix)
println("Accuracy: " + metrics.accuracy) 
println(s"Test Error = ${(1.0 - metrics.accuracy)}")


/*
// 10. Set the maximum number of iterations and the regularization parameter .
val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)

// 11. Make a fit to adjust the model.
val lsvcModel = lsvc.fit(dataIndexed)

// 12. Print the coefficients and intercepts for the Linear SVC.
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
*/
