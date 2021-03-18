// Fibonacci Sequence
// Case 1: Recursion - Pattern Matching
// – Simple and well suited for small numbers, but it doesn’t scale
// – If n is big, we run the risk of getting a Stack Overflow
def fibRecursion(n: Long): Long = n match {
  case 0 | 1 => n
  case _ => fib1(n - 1) + fib1(n - 2)
}


// Case 2: Loop
// – Handles Long numbers (64 bit)
// – A little bit too verbose, non-idiomatic, mutable variables.
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

// Case 3: Tail Recursion
// – Optimized by the compiler. We say a function is tail recursive when the recursive call is the last thing executed by the function. 
// We can see the fib_tail call being applied in the last line of code.
def fibTailRecursion(n:Int): Int = {
    def fib_tail(n: Int,a: Int, b: Int): Int = n match {
        case 0 => a
        case _=> fib_tail(n - 1, b, a + b)
    }
    return fib_tail(n, 0, 1)
}


/*
   * Case 4: Memoization
   * - Substitute number in variable "s" with amount of numbers to print
   * - Not suitable for big amounts (190 max approx.)
   */
  val fib: Stream[BigInt] = 0 #:: 1 #:: fib.zip(fib.tail).map(p => p._1 + p._2)

  def main(args: Array[String]){
    val s = fib take 135 mkString " "
    print(s)
    println()
    print(fib(130))
  }

 /*
   * Case 5 Pisano period
   * - Get last 6 digits of Fibonacci with tail recursion
   */
  
  def fib5( n : Int) : Int = { 
    def fib_tail( n: Int, a:Int, b:Int): Int = n match {
      case 0 => a 
      case _ => fib_tail( n-1, b, (a+b)%1000000 )
    }
    return fib_tail( n%1500000, 0, 1)
  }
}


//Case 6: divide and conquer
 
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

