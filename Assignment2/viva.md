* Binary Classifier on different pixel sizes:
```
Validation Set:
16 -> 71.750 , 77.750
32 -> 71.000 , 77.750
64 -> 64.000 , 77.500
128 -> 62.000, 78.250

Training Data:
16 -> 88.172 , 77.521
32 ->  
64 -> 99.895, 86.786  
128 -> 


```

* Gaussiaa Naive Bayes:
- Naive bayes but now with continuous parametrs: you should p(x_i | y) is now obtaines from a normal distribution.
- Each feature is assumed normally distributed and imdependently
- The paramaters then for each feature can be leaned

* Naive Bayes for case x = vectro of binary values. Reduces to the logistic regression form
* Coordinate ascent:
  - Fix (n-1) parametes. Maximise w.r.t. one. Do this iteatively cycling w.r.t. the parameter we're maximising against
  - Why does it work?
  - We can always produce an example where it does not work
  - Assume we have a concave function and hence just 1 maxima
  - Notice thst in each step, the objective value strictly increases
  - So we could expect to reach the global maxima iteratively. Guaranteed Increast
  - Actually it's bound to get closer (in the objective value sense not in (x1 ,x2) distance sense) to the maxima in the concave case

* SMO Algorithm:
  - choose 2 $\alpha$'s then, do coordinte ascent by keeping others fixed
  - Objective is Quadratic w.r.t $alpa_2$, given $\alpha_2$ we get $\alpha_1$ 
  - "clipping" for box constraint
  - Heuristics to choose which 2 $\alpha$'s to pick 
* 
