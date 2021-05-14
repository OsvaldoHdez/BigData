// Practice 2
// 1. Create a list called "list" with the elements "red", "white", "black"
 var list1 = List("Red", "White", "Black");
 println(list1);

// 2. Add 5 more elements to "list" "green", "yellow", "blue", "orange", "pearl"
 list1 = list1 :+ "Green" :+ "Yellow" :+ "Blue" :+ "Orange" :+ "Pearl";

// 3. Get the elements of "list" "green", "yellow", "blue"
 list1.slice(3,6);

// 4. Create an array of numbers in the range 1-1000 in steps of 5 by 5
 val array1 = Array.range(1,1000,5); //or range 0-1000

// 5. What are the unique elements of the list List (1,3,3,4,6,7,3,7) use conversion to sets
 scala> var Lista2 = List(1,3,3,4,6,7,3,7)
 Lista2: List[Int] = List(1, 3, 3, 4, 6, 7, 3, 7)

 scala> Lista2.toSet
 res8: scala.collection.immutable.Set[Int] = Set(1, 6, 7, 3, 4)

// 6. Create a mutable map named names that contains the following "Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27"
 scala> val mutablemap = collection.mutable.Map(("Jose", 20), ("Luis", 24), ("Ana", 23), ("Susana", 27))
 mutablemap: scala.collection.mutable.Map[String,Int] = Map(Susana -> 27, Ana -> 23, Luis -> 24, Jose -> 20)

// 6 a. Print all keys on the map
 scala> mutablemap.keys
 res9: Iterable[String] = Set(Susana, Ana, Luis, Jose)

// 6 b. Add the following value to the map ("Miguel", 23) 
 scala> mutablemap += ("Miguel" -> 23)
 res10: mutablemap.type = Map(Susana -> 27, Ana -> 23, Miguel -> 23, Luis -> 24, Jose -> 20)
