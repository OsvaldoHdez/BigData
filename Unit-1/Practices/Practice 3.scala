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