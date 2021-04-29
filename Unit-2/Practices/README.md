# Practices
---
## Practice 1 - Basic Statistics
### Correlation
Calculating the correlation between two series of data is a common operation in Statistics. In spark.ml we provide the flexibility to calculate pairwise correlations among many series. The supported correlation methods are currently Pearson’s and Spearman’s correlation.
#### Code
```scala
// 1. Importing the library for matrices and vectors 
import org.apache.spark.ml.linalg.{Matrix, Vectors}

// 2. Import correlation library 
import org.apache.spark.ml.stat.Correlation

// 3. Allows access to a single row value through generic ordinal access, as well as primitive access
import org.apache.spark.sql.Row

// 4. Session import and creation 
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("CorrelationExample").getOrCreate()

// 5. Importing the Implicit Conversion to Convert RDDs to DataFrames 
import spark.implicits._

// 6. Creation of dense and sparse vectors, within a matrix
val data = Seq(
    Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))),
    Vectors.dense(4.0, 5.0, 0.0, 3.0),
    Vectors.dense(6.0, 7.0, 0.0, 8.0),
    Vectors.sparse(4, Seq((0, 9.0), (3, 1.0)))
)

// 7. The data is extracted from the matrix and a dataframe is created 
val df = data.map(Tuple1.apply).toDF("features")

// 8. Creation of the Pearson correlation matrix using the created dataframe 
val Row(coeff1: Matrix) = Correlation.corr(df, "features").head

// 9. Print result 
println(s"Pearson correlation matrix:\n $coeff1")

// 10. Creating the Spearman correlation matrix using the created dataframe 
val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head

// 11. Print result 
println(s"Spearman correlation matrix:\n $coeff2")
```
#### Results
```scala
// 1. Importing the library for matrices and vectors 
scala> import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.ml.linalg.{Matrix, Vectors}

// 2. Import correlation library 
scala> import org.apache.spark.ml.stat.Correlation
import org.apache.spark.ml.stat.Correlation

// 3. Allows access to a single row value through generic ordinal access, as well as primitive access
scala> import org.apache.spark.sql.Row
import org.apache.spark.sql.Row

// 4. Session import and creation 
scala> import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SparkSession
scala> val spark = SparkSession.builder.appName("CorrelationExample").getOrCreate()
21/04/29 14:40:51 WARN SparkSession$Builder: Using an existing SparkSession; some spark core configurations may not take effect.
spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@77a14911

// 5. Importing the Implicit Conversion to Convert RDDs to DataFrames 
scala> import spark.implicits._
import spark.implicits._

// 6. Creation of dense and sparse vectors, within a matrix
scala> val data = Seq(
     |     Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))),
     |     Vectors.dense(4.0, 5.0, 0.0, 3.0),
     |     Vectors.dense(6.0, 7.0, 0.0, 8.0),
     |     Vectors.sparse(4, Seq((0, 9.0), (3, 1.0)))
     | )
data: Seq[org.apache.spark.ml.linalg.Vector] = List((4,[0,3],[1.0,-2.0]), [4.0,5.0,0.0,3.0], [6.0,7.0,0.0,8.0], (4,[0,3],[9.0,1.0]))

// 7. The data is extracted from the matrix and a dataframe is created 
scala> val df = data.map(Tuple1.apply).toDF("features")
df: org.apache.spark.sql.DataFrame = [features: vector]

// 8. Creation of the Pearson correlation matrix using the created dataframe 
scala> val Row(coeff1: Matrix) = Correlation.corr(df, "features").head
21/04/29 14:42:03 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
21/04/29 14:42:03 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
21/04/29 14:42:04 WARN PearsonCorrelation: Pearson correlation matrix contains NaN values.
coeff1: org.apache.spark.ml.linalg.Matrix =
1.0                   0.055641488407465814  NaN  0.4004714203168137
0.055641488407465814  1.0                   NaN  0.9135958615342522
NaN                   NaN                   1.0  NaN
0.4004714203168137    0.9135958615342522    NaN  1.0

// 9. Print result 
scala> println(s"Pearson correlation matrix:\n $coeff1")
Pearson correlation matrix:
 1.0                   0.055641488407465814  NaN  0.4004714203168137  
0.055641488407465814  1.0                   NaN  0.9135958615342522  
NaN                   NaN                   1.0  NaN                 
0.4004714203168137    0.9135958615342522    NaN  1.0   

// 10. Creating the Spearman correlation matrix using the created dataframe 
scala> val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head
21/04/29 14:43:34 WARN PearsonCorrelation: Pearson correlation matrix contains NaN values.
coeff2: org.apache.spark.ml.linalg.Matrix =
1.0                  0.10540925533894532  NaN  0.40000000000000174
0.10540925533894532  1.0                  NaN  0.9486832980505141
NaN                  NaN                  1.0  NaN
0.40000000000000174  0.9486832980505141   NaN  1.0

// 11. Print result 
scala> println(s"Spearman correlation matrix:\n $coeff2")
Spearman correlation matrix:
 1.0                  0.10540925533894532  NaN  0.40000000000000174  
0.10540925533894532  1.0                  NaN  0.9486832980505141   
NaN                  NaN                  1.0  NaN                  
0.40000000000000174  0.9486832980505141   NaN  1.0 
```
### Hypothesis testing
Hypothesis testing is a powerful tool in statistics to determine whether a result is statistically significant, whether this result occurred by chance or not. spark.ml currently supports Pearson’s Chi-squared (χ2) tests for independence.

ChiSquareTest conducts Pearson’s independence test for every feature against the label. For each feature, the (feature, label) pairs are converted into a contingency matrix for which the Chi-squared statistic is computed. All label and feature values must be categorical.
#### Code
```scala
// 1. Importing the library for vectors
import org.apache.spark.ml.linalg.{Vector, Vectors}

// 2. ChiSquare library import 
import org.apache.spark.ml.stat.ChiSquareTest

// 3. Session import and creation 
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("ChiSquareTestExample").getOrCreate()

// 4. Importing the Implicit Conversion to Convert RDDs to DataFrames 
import spark.implicits._

// 5. Create dense vectors 
val data = Seq(
    (0.0, Vectors.dense(0.5, 10.0)),
    (0.0, Vectors.dense(1.5, 20.0)),
    (1.0, Vectors.dense(1.5, 30.0)),
    (0.0, Vectors.dense(3.5, 30.0)),
    (0.0, Vectors.dense(3.5, 40.0)),
    (1.0, Vectors.dense(3.5, 40.0))
)

// 6. Creation of the dataframe with the previous data 
val df = data.toDF("label", "features")

// 7. The first values of the dataframe are taken 
val chi = ChiSquareTest.test(df, "features", "label").head
```
#### Results
```scala
// 1. Importing the library for vectors
scala> import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.linalg.{Vector, Vectors}

// 2. ChiSquare library import 
scala> import org.apache.spark.ml.stat.ChiSquareTest
import org.apache.spark.ml.stat.ChiSquareTest

// 3. Session import and creation 
scala> import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SparkSession
scala> val spark = SparkSession.builder.appName("ChiSquareTestExample").getOrCreate()
21/04/29 15:00:00 WARN SparkSession$Builder: Using an existing SparkSession; some spark core configurations may not take effect.
spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@7e2c8622

// 4. Importing the Implicit Conversion to Convert RDDs to DataFrames 
scala> import spark.implicits._
import spark.implicits._

// 5. Create dense vectors 
scala> val data = Seq(
     |     (0.0, Vectors.dense(0.5, 10.0)),
     |     (0.0, Vectors.dense(1.5, 20.0)),
     |     (1.0, Vectors.dense(1.5, 30.0)),
     |     (0.0, Vectors.dense(3.5, 30.0)),
     |     (0.0, Vectors.dense(3.5, 40.0)),
     |     (1.0, Vectors.dense(3.5, 40.0))
     | )
data: Seq[(Double, org.apache.spark.ml.linalg.Vector)] = List((0.0,[0.5,10.0]), (0.0,[1.5,20.0]), (1.0,[1.5,30.0]), (0.0,[3.5,30.0]), (0.0,[3.5,40.0]), (1.0,[3.5,40.0]))

// 6. Creation of the dataframe with the previous data 
scala> val df = data.toDF("label", "features")
df: org.apache.spark.sql.DataFrame = [label: double, features: vector]

// 7. The first values of the dataframe are taken 
scala> val chi = ChiSquareTest.test(df, "features", "label").head
chi: org.apache.spark.sql.Row = [[0.6872892787909721,0.6822703303362126],WrappedArray(2, 3),[0.75,1.5]]

```
