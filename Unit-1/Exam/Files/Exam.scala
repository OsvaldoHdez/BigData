// 1
import org.apache.spark.sql.SparkSession
val spar = SparkSession.builder()

// 2
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Unit-1/Exam/Files/Netflix_Stock.csv")

// 3
df.show() 

// 4
df.printSchema()

// 5
df.head(5)

// 6
df.describe().show()

// 7
val df2 = df.withColumn("HV Ratio", df("High")*df("Volume"))

// 8


// 9

// 10

// 11

