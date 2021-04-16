// 1. Start a Spark sesion 
import org.apache.spark.sql.SparkSession
val spar = SparkSession.builder()

// 2. load the netflix stock CSV, make spark infiere all the data
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Unit-1/Exam/Files/Netflix_Stock.csv")

// 3. wich are the names of the columns
df.show() 

// 4. how is the schema
df.printSchema()

// 5. print the 1st 5 columns
df.head(5)

// 6. use describe () to understand the DataFrame
df.describe().show()

// 7. create a new dataframe with a new column called "HV Ratio" that is the relacion between the price on the "High" column in relation to the "Volume" of negociated stocks for one day
val df2 = df.withColumn("HV Ratio", df("High")*df("Volume"))
df2.show() 

// 8. wich day had the highest peek on the "Close" column
import spark.implicits._
df.select(max("Close")).show()
df.filter("Close = 707.610001 ").show()

// 9. In your own words write a comment on your code. 
// What is the meaning of the column "Close"?
// Is the number or amount that the stocks of netflix ended on that day

// 10. what is the minimum and maximum of "Volume"
df.select(min("Volume")).show()
df.select(max("Volume")).show()

// 11. With Scala / Spark $ Syntax, answer the following: 
val df3 = df
// a. ¿How many days was the “Close” column less than $ 600? 
df3.filter("Close<600").count()

// b. ¿What percentage of the time was the “High” column greater than $ 500? 
(df3.filter("High>500").count()*1.0/df3.count())*100

// c. ¿What is the Pearson correlation between column "High" and column "Volume"? 
df3.select(corr("High", "Close")).show()

// d. ¿What is the maximum in the “High” column per year?
val years = df3.withColumn("Year",year(df3("Date")))
years.select("Year", "High").groupBy("Year").max("High").show()

// e. ¿What is the “Close” column average for each calendar month? 
val months = df3.withColumn("Month", month(df3("Date")))
months.select("Month", "Close").groupBy("Month").mean("Close").show()
