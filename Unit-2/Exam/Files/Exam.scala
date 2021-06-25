// Se importan las librerías
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline

// Se importa la sesión
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("MultilayerPerceptronClassifierExample").getOrCreate()

// 1. Cargar en un dataframe Iris.csv
val data = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("/iris.csv")
val dataClean = data.na.drop()

// 2. ¿Cuáles son los nombres de las columnas?
dataClean.columns

// 3. ¿Cómo es el esquema?
dataClean.printSchema()

// 4. Imprime las primeras 5 columnas.
dataClean.show(5)

// 5. Usa el metodo describe () para aprender mas sobre los datos del DataFrame.
dataClean.describe().show()

// 6. Haga la transformación pertinente para los datos categoricos los cuales serán nuestras etiquetas a clasificar.
    // Se declara un vector que transformara los datos a la variable "features"
    val vectorFeatures = (new VectorAssembler().setInputCols(Array("sepal_length","sepal_width", "petal_length","petal_width")).setOutputCol("features"))

    // Se transforman los features usando el dataframe
    val features = vectorFeatures.transform(dataClean)

    // Se declara un "StringIndexer" que transformada los datos de "species" en datos numericos
    val speciesIndexer = new StringIndexer().setInputCol("species").setOutputCol("label")

    // Ajustamos las especies indexadas con el vector features
    val dataIndexed = speciesIndexer.fit(features).transform(features)

// 7. Construya el modelo de clasificación y explique su arquitectura.
    // Con la variable "splits" hacemos un corte de forma aleatoria
    val splits = dataIndexed.randomSplit(Array(0.6, 0.4), seed = 1234L)

    // Se declara la variable "train" la cual tendra el 60% de los datos
    val train = splits(0)

    // Se declara la variable "test" la cual tendra el 40% de los datos
    val test = splits(1)

    // Se establece la configuración de las capas para el modelo de redes neuronales artificiales
    // De entrada: 4 (features)
    // las dos Intermedias  5 y 4 respectivamente
    // de salida 3, al tener 3 tipos de clases
    val layers = Array[Int](4, 5, 4, 3)

    // Se configura el entrenador del algoritmo Multilayer con sus respectivos parametros
    val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

    // Se entrena el modelo con los datos de entrenamiento
    val model = trainer.fit(train)

    // Se prueban ya entrenado el modelo
    val result = model.transform(test)

// 8. Imprima los resultados del modelo
    // Se selecciona la predicción y la etiqueta que seran guardado en la variable
    val predictionAndLabels = result.select("prediction", "label")

    // Se muestran algunos datos de la predicción contra los reales para ver resultados
    predictionAndLabels.show(50)
    result.show(30)

    // Se ejecuta la estimación de la precisión del modelo
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
    println(s"Prueba de precision = ${evaluator.evaluate(predictionAndLabels)}")
