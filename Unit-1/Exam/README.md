# Exam U1
### 1. Start a Spark sesion 
#### Code
```scala
import org.apache.spark.sql.SparkSession
val spar = SparkSession.builder()
```
#### Results
<html><div align="center"><img src="https://i.ibb.co/NZFmmNJ/imagen.png"></div></html>

### 2. Load the netflix stock CSV, make spark infiere all the data
#### Code
```scala
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Unit-1/Exam/Files/Netflix_Stock.csv")
```
#### Results
<html><div align="center"><img src="https://i.ibb.co/pzMdQSY/imagen.png"></div></html>

### 3. ¿Wich are the names of the columns?
#### Code
```scala
df.show() 
```
#### Results
<html><div align="center"><img src="https://i.ibb.co/R7PjDTD/imagen.png"></div></html>


### 4. ¿How is the schema?
#### Code
```scala
df.printSchema()
```
#### Results
<html><div align="center"><img src="https://i.ibb.co/3W0tD9X/imagen.png"></div></html>

### 5. Print the 1st 5 columns
#### Code
```scala
df.head(5)
```
#### Results
<html><div align="center"><img src="https://i.ibb.co/sFmpM5j/imagen.png"></div></html>

### 6. Use describe () to understand the DataFrame
#### Code
```scala
df.describe().show()
```
#### Results
<html><div align="center"><img src="https://i.ibb.co/12QcrKS/imagen.png"></div></html>

### 7. Create a new dataframe with a new column called "HV Ratio" that is the relacion between the price on the "High" column in relation to the "Volume" of negociated stocks for one day
#### Code
```scala
val df2 = df.withColumn("HV Ratio", df("High")*df("Volume"))
```
#### Results
<html><div align="center"><img src="https://i.ibb.co/6X01dwf/imagen.png"></div></html>

### 8. ¿Which day had the highest peek on the "Close" column?
#### Code
```scala
import spark.implicits._

df.select(max("Close")).show()
df.filter("Close = 707.610001 ").show()
```
#### Results
<html><div align="center"><img src="https://i.ibb.co/M5pkv4k/imagen.png"></div></html>

### 9. In your own words write a comment on your code. 
#### ¿What is the meaning of the column "Close"?
Is the number or amount that the stocks of netflix ended on that day

### 10. ¿What is the minimum and maximum of "Volume"?
#### Code
```scala
df.select(min("Volume")).show()
df.select(max("Volume")).show()
```
#### Results
<html><div align="center"><img src="https://i.ibb.co/nCgW74g/imagen.png"></div></html>

### 11. With Scala / Spark $ Syntax, answer the following:
Hint: Basically very similar to the dates session, you will have to create another dataframe to answer some of the items.
<html><div align="center"><img src="https://i.ibb.co/JFNps34/imagen.png"></div></html>

1. ¿How many days was the “Close” column less than $ 600? 
    #### Code
    ```scala
    df3.filter("Close<600").count()

    ```
    #### Results
    <html><div align="center"><img src="https://i.ibb.co/cY3Sn7F/imagen.png"></div></html>

2. ¿What percentage of the time was the “High” column greater than $ 500?
    #### Code
    ```scala
    (df3.filter("High>500").count()*1.0/df3.count())*100
    ```
    #### Results
    <html><div align="center"><img src="https://i.ibb.co/qFLFPss/imagen.png"></div></html>

3. ¿What is the Pearson correlation between column "High" and column "Volume"?
    #### Code
    ```scala
    df3.select(corr("High", "Close")).show()
    ```
    #### Results
    <html><div align="center"><img src="https://i.ibb.co/tDgKcQq/imagen.png"></div></html>

4. ¿What is the maximum in the “High” column per year?
    #### Code
    ```scala
    val years = df3.withColumn("Year",year(df3("Date")))
    years.select("Year", "High").groupBy("Year").max("High").show()

    ```
    #### Results
    <html><div align="center"><img src="https://i.ibb.co/LCZpjyj/imagen.png"></div></html>
    
5. ¿What is the “Close” column average for each calendar month? 
    #### Code
    ```scala
    val months = df3.withColumn("Month", month(df3("Date")))
    months.select("Month", "Close").groupBy("Month").mean("Close").show()
    ```
    #### Results
    <html><div align="center"><img src="https://i.ibb.co/Tktw2Vv/imagen.png"></div></html>


