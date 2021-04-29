// Correlación

// Importación de la librería de para matrices y vectores
import org.apache.spark.ml.linalg.{Matrix, Vectors}

// Importación de la librería de correlación
import org.apache.spark.ml.stat.Correlation

// Permite acceder a un valor de una fila a través del acceso genérico por ordinal,  así como el acceso primitivo.
import org.apache.spark.sql.Row

// Importación y creación de la sesión
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("CorrelationExample").getOrCreate()

// Importación de la conversión implícita para convertir RDDs a DataFrames 
import spark.implicits._

// Creación de vectores densos y dispersos, dentro de una matriz
val data = Seq(
    Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))),
    Vectors.dense(4.0, 5.0, 0.0, 3.0),
    Vectors.dense(6.0, 7.0, 0.0, 8.0),
    Vectors.sparse(4, Seq((0, 9.0), (3, 1.0)))
)

// Se extraen los datos de la matriz y se crea un dataframe
val df = data.map(Tuple1.apply).toDF("features")

// Creación de la matriz de correlación de Pearson usando el dataframe creado
val Row(coeff1: Matrix) = Correlation.corr(df, "features").head

// Imprimir resultado
println(s"Pearson correlation matrix:\n $coeff1")

// Creación de la matriz de correlación de Spearman usando el dataframe creado
val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head

// Imprimir resultado
println(s"Spearman correlation matrix:\n $coeff2")

//Evaluación de la hipótesis 

// Importación de la librería de para vectores
import org.apache.spark.ml.linalg.{Vector, Vectors}

// Importación de la librería ChiSquare
import org.apache.spark.ml.stat.ChiSquareTest

// Importación y creación de la sesión
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("ChiSquareTestExample").getOrCreate()

// Importación la conversión implícita para convertir RDDs a DataFrames 
import spark.implicits._

// Creación de vectores densos
val data = Seq(
    (0.0, Vectors.dense(0.5, 10.0)),
    (0.0, Vectors.dense(1.5, 20.0)),
    (1.0, Vectors.dense(1.5, 30.0)),
    (0.0, Vectors.dense(3.5, 30.0)),
    (0.0, Vectors.dense(3.5, 40.0)),
    (1.0, Vectors.dense(3.5, 40.0))
)

// Creación del dataframe con los datos anteriores
val df = data.toDF("label", "features")

// Se toman los primeros valores del dataframe
val chi = ChiSquareTest.test(df, "features", "label").head
