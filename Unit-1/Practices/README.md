# Practices
---
## Practice 1
#### 1. Develop a scala algorithm that calculates the radius of a circle.
```scala
val pi: Double = 3.1416;
var perimetro: Double = 17.2788;
println(perimetro / (2*pi))
```
#### 2. Develop a scala algorithm that tells me if a number is prime.
```scala
// Method 1
def isPrime(i: Int): Boolean ={
    if (i <= 1)
        false
    else if (i == 2)
        true
    else
        !(2 until i).exists(n => i % n == 0)
}
// Method 2
def isPrime1(n: Int): Boolean = ! ((2 until n-1) exists (n % _ == 0)) 
```
#### 3. Given the variable bird = "tweet", use string interpolation to print "I'm writing a tweet".
```scala
val bird = "tweet"
println(s"I'm writing a $bird")
```
#### 4. Given the variable message = "Hello Luke, I am your father!" uses slilce to extract the sequence "Luke"
```scala
 scala> val mensaje = "Hello Luke I am your father!"
 mensaje: String = Hola Luke yo soy tu padre!

 scala> mensaje.slice (5,9)
 res2: String = Luke
```
#### 5. What is the difference between value and a variable in scala?
```scala
The variable val can NOT be modified once its value has been predefined.

The variable var YES its values can be overwritten, it can be modified. (It can only be overwritten if the data to be entered is the same type of data as the previous one)
```
#### 6. Given the tuple (2,4,5,1,2,3,3.1416,23) returns the number 3.1416

```scala
scala> val tupla = (2,4,5,1,2,3,3.1416,23)
tupla: (Int, Int, Int, Int, Int, Int, Double, Int) = (2,4,5,1,2,3,3.1416,23)

scala> tupla._7
res25: Double = 3.1416
```
---
## Practice 2
#### 1. Create a list called "list" with the elements "red", "white", "black".
```scala
var list = List("Red", "White", "Black");
println(list);
```
#### 2. Add 5 more elements to "list" "green", "yellow", "blue", "orange", "pearl".
```scala
list = list :+ "Green" :+ "Yellow" :+ "Blue" :+ "Orange" :+ "Pearl";
```
#### 3. Get the elements of "list" "green", "yellow", "blue".
```scala
list.slice(3,6);
``` 
#### 4. Create an array of numbers in the range 1-1000 in steps of 5 by 5.
```scala
val array1 = Array.range(1,1000,5); //or range 0-1000
``` 

#### 5. What are the unique elements of the list List (1,3,3,4,6,7,3,7) use conversion to sets
 
```scala
scala> var Lista2 = List(1,3,3,4,6,7,3,7)
 Lista2: List[Int] = List(1, 3, 3, 4, 6, 7, 3, 7)
 
  scala> Lista2.toSet
 res8: scala.collection.immutable.Set[Int] = Set(1, 6, 7, 3, 4)
``` 

#### 6. Create a mutable map named names that contains the following "Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27"
```scala
scala> val mutablemap = collection.mutable.Map(("Jose", 20), ("Luis", 24), ("Ana", 23), ("Susana", 27))
 mutablemap: scala.collection.mutable.Map[String,Int] = Map(Susana -> 27, Ana -> 23, Luis -> 24, Jose -> 20)
``` 
 #### 6 a. Print all keys on the map

```scala scala> mutablemap.keys
 res9: Iterable[String] = Set(Susana, Ana, Luis, Jose)
``` 
#### 6 b. Add the following value to the map ("Miguel", 23) 
```scala 
 scala> mutablemap += ("Miguel" -> 23)
 res10: mutablemap.type = Map(Susana -> 27, Ana -> 23, Miguel -> 23, Luis -> 24, Jose -> 20)
``` 


---
## Practice 3 - Fibonacci
#### Case 1. Recursion - Pattern Matching
- Simple and well suited for small numbers, but it doesn’t scale.
- If 'n' is big, we run the risk of getting a Stack Overflow.
```scala
def fibRecursion(n: Long): Long = n match {
  case 0 | 1 => n
  case _ => fib1(n - 1) + fib1(n - 2)
}
``` 
#### Case 2. Loop
- Handles Long numbers (64 bit).
- A little bit too verbose, non-idiomatic, mutable variables.
```scala
def fibLoop(n:Long):Long = {
    var first = 0;
    var second = 1;
    var count = 0;
    while(count < n){
        val sum = first + second
        first = second
        second = sum
        count = count + 1
    }
    return first
}
```
#### Case 3. Tail Recursion
- Optimized by the compiler. We say a function is tail recursive when the recursive call is the last thing executed by the function. 
- The fib_tail call being applied in the last line of code.
```scala
def fibTailRecursion(n:Int): Int = {
    def fib_tail(n: Int,a: Int, b: Int): Int = n match {
        case 0 => a
        case _=> fib_tail(n - 1, b, a + b)
    }
    return fib_tail(n, 0, 1)
}
```
#### Case 4: Memoization
- Substitute number in variable "s" with amount of numbers to print
- Not suitable for big amounts (190 max approx.)
 ```scala
  val fib: Stream[BigInt] = 0 #:: 1 #:: fib.zip(fib.tail).map(p => p._1 + p._2)

  def main(args: Array[String]){
    val s = fib take 135 mkString " "
    print(s)
    println()
    print(fib(130))
  }
 ```
#### Case 5 Pisano period   
- Get last 6 digits of Fibonacci with tail recursion
```scala
    def fib5( n : Int) : Int = { 
    def fib_tail( n: Int, a:Int, b:Int): Int = n match {
      case 0 => a 
      case _ => fib_tail( n-1, b, (a+b)%1000000 )
    }
    return fib_tail( n%1500000, 0, 1)
  }
}
```
#### Case 6: divide and conquer
```scala
def fib6(n :int):Double={
    if(n<=0){
        return 0
    }
 
    var i = n-1
    var aux1 =0.0
    var aux2=1.0
 
    var ab=(aux2,aux1)
    var cd=(aux1,aux2)
 
    while(i>0){
        if((i%2)!=0){
            aux1=(cd._2*ab._2)+(cd._1*ab._1)
            aux2=(cd._2*(ab._2+ab._1)+cd._1*ab._2)
            ab=(aux1,aux2)
        }
 
        aux1=pow(cd._1,2)+pow(cd._2,2)
        aux2=(cd._2*(2*cd._1+cd._2))
        cd=(aux1,aux2)
        i=i/2
    }
    return (ab._1+ab._2)
 
}
fib6(10)
```

  
 
 

