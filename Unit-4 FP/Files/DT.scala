// 1. Import libraries
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}

// 2. Import session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("DecisionTreeClassificationExample").getOrCreate()

// 3. Load the data stored in LIBSVM format as a DataFrame.
val data  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("/home/valdo/Documentos/Gitkraken/BigData/Unit-4 FP/Files/bank-full.csv")

// 4. Process of categorizing the variables type string to numeric. 
val yes = data.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
val no = yes.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
val newcolumn = no.withColumn("y",'y.cast("Int"))

// 5. Vector is created with the column features 
val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")

// 6. Transforms into a new df 
val data2 = assembler.transform(newcolumn)

// 7. Column and label are given a new name 
val featuresLabel = data2.withColumnRenamed("y", "label")

// 8. Select index
val dataIndexed = featuresLabel.select("label","features")
// Index columns

// 9. Index labels, adding metadata to the label column.
//    Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(dataIndexed)

// 10. Automatically identify categorical features, and index them.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(dataIndexed)
// features with > 4 distinct values are treated as continuous.

// 11. Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = dataIndexed.randomSplit(Array(0.7, 0.3))

// 12. Train a DecisionTree model.
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

// 13. Convert indexed labels back to original labels.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// 14. Chain indexers and tree in a Pipeline.
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

// 15. Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

// 16. Make predictions.
val predictions = model.transform(testData)

// 17. Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// 18. Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")

