// 1. Import libraries
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.{StringIndexer, VectorIndexer, VectorAssembler}

// 2. Import session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("MultilayerPerceptronClassifierExample").getOrCreate()

// 3. Load the data stored in LIBSVM format as a DataFrame.
val data  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("/Files/bank-full.csv")

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

// 9. Split the data into train and test
val splits = dataIndexed.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)

// 10. Specify layers for the neural network:
//    input layer of size 5 (features), two intermediate of size 2 and 2
//    and output of size 4 (classes)
val layers = Array[Int](5,2,2,4)

// 11. Create the trainer and set its parameters
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

// 12. Train the model
val model = trainer.fit(train)

// 13. Compute accuracy on the test set
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")