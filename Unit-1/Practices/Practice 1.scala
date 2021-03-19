// Assessment 1 / Practice 1
//1. Develop a scala algorithm that calculates the radius of a circle
val pi: Double = 3.1416;
var perimetro: Double = 17.2788;
println(perimetro / (2*pi))

//2. Develop a scala algorithm that tells me if a number is prime
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


//3. Given the variable bird = "tweet", use string interpolation to print "I'm writing a tweet"
val bird = "tweet"
println(s"I'm writing a $bird")

//4. Given the variable message = "Hello Luke, I am your father!" uses slilce to extract the sequence "Luke"
 scala> val mensaje = "Hello Luke I am your father!"
 mensaje: String = Hola Luke yo soy tu padre!

 scala> mensaje.slice (5,9)
 res2: String = Luke

//5. What is the difference between value and a variable in scala?
The variable val can NOT be modified once its value has been predefined.

The variable var YES its values can be overwritten, it can be modified. (It can only be overwritten if the data to be entered is the same type of data as the previous one)
 
//6. Given the tuple (2,4,5,1,2,3,3.1416,23) returns the number 3.1416

scala> val tupla = (2,4,5,1,2,3,3.1416,23)
tupla: (Int, Int, Int, Int, Int, Int, Double, Int) = (2,4,5,1,2,3,3.1416,23)

scala> tupla._7
res25: Double = 3.1416
