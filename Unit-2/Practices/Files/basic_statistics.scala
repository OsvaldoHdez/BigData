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

rintln(s"pValues = ${chi.getAs[Vector](0)}")
//Despues se buscaran los grados de libertad del modelo
println(s"degreesOfFreedom ${chi.getSeq[Int](1).mkString("[", ",", "]")}")
//por ultimo se extraeran ciertos valores de un vector dererminado todo en base a la funcion chi cuadrado
println(s"statistics ${chi.getAs[Vector](2)}")


//Summarizer

//importacion de librerias necesarias, en este uso de vectores y el propio summarizer
import spark.implicits._    
import Summarizer._

//se crea un conjunto de vectores o secuencia
val data = Seq(
  (Vectors.dense(2.0, 3.0, 5.0), 1.0),
  (Vectors.dense(4.0, 6.0, 7.0), 2.0)
)

//Creacion del dataframe a partir de los vectores
val df = data.toDF("features", "weight")


//se hace uso de la libreria summarizer para obtener la media y la varianza de algunos datos en el dataframe solicitado
val (meanVal, varianceVal) = df.select(metrics("mean", "variance").summary($"features", $"weight").as("summary")).select("summary.mean", "summary.variance").as[(Vector, Vector)].first()

//se imprimen las variables trabajadas anteriormente
println(s"with weight: mean = ${meanVal}, variance = ${varianceVal}")

//se repite el procesos con 2 nuevas variables 
val (meanVal2, varianceVal2) = df.select(mean($"features"), variance($"features"))
  .as[(Vector, Vector)].first()
// impresión de variables
println(s"without weight: mean = ${meanVal2}, sum = ${varianceVal2}")

