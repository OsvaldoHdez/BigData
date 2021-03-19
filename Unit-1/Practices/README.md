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
---
## Practice 3 - Fibonacci
#### 1. Recursion - Pattern Matching
- Simple and well suited for small numbers, but it doesn’t scale.
- If 'n' is big, we run the risk of getting a Stack Overflow.
```scala
def fibRecursion(n: Long): Long = n match {
  case 0 | 1 => n
  case _ => fib1(n - 1) + fib1(n - 2)
}
``` 
#### 2. Loop
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
#### 3. Tail Recursion
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