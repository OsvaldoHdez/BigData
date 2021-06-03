// Import the "LinearSVC" library, this binary classifier optimizes the hinge loss using the OWLQN optimizer. 
import org.apache.spark.ml.classification.LinearSVC

// Import session.
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("LinearSVCExample").getOrCreate()

// Load the training data. 
val training = spark.read.format("libsvm").load("/Files/sample_libsvm_data.txt")

// Set the maximum number of iterations and the regularization parameter .
val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)

// Make a fit to adjust the model.
val lsvcModel = lsvc.fit(training)

// Print the coefficients and intercepts for the Linear SVC.
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")