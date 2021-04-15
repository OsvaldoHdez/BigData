# Exams
---
## Exam 1

## 1 Start a Spark sesion 
```scala
import org.apache.spark.sql.SparkSession
val spar = SparkSession.builder()
```

## 2 load the netflix stock CSV, make spark infiere all the data
```scala
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Unit-1/Exam/Files/Netflix_Stock.csv")
```



## 3 wich are the names of the columns
```scala
df.show() 

```

##  4 how is the schema
```scala
df.printSchema()
```


##  5 print the 1st 5 columns
```scala
df.head(5)
```


###  6 use describe () to understand the DataFrame
```scala
df.describe().show()
```


###  7 create a new dataframe with a new column called "HV Ratio" that is the relacion between the price on the "High" column in relation to the "Volume" of negociated stocks for one day
```scala
val df2 = df.withColumn("HV Ratio", df("High")*df("Volume"))
```


###  8 wich day had the highest peek on the "Close" column
```scala
import spark.implicits._

df.select(max("Close")).show()
df.filter("Close = 707.610001 ").show()

```


###  9 In your own words write a comment on your code. 
#### What is the meaning of the column "Close"
Is the number or amount that the stocks of netflix ended on that day


### 10 what is the minimum and maximum of "Volume"
```scala
df.select(min("Volume")).show()
df.select(max("Volume")).show()

```
