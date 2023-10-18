* Binary Classifier on different pixel sizes:
```
Image Size: --> Accuracy by Lineat Kernel and Gaussian Kernel Respectively (SK Learn)
Validation Set:
16 ->  71.750 , 77.750
32 ->  71.000 , 77.750
64 ->  64.000 , 77.500
128 -> 62.000 , 78.250

Training Data:
16 ->  78.046 , 73.361
32 ->  88.172 , 77.521 
64 ->  99.895 , 86.786  
128 -> 99.979 , 97.206 
```

* Above clearly indicates, the model just iverfits as you make the image "fed into the SVM classfier" more clearer
* Which makes sence, two very close pixels, don't give out much info (for an SVM classfier), differnece in their value is just noise

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
* Multiclass Classfier from Binary
  - One vs One: done in assignment
  - One vs Many:

* Multiclass Imbalance: In multiclass classification, class imbalance can occur when some classes have very few instances while others have a larger number. The degree of imbalance can vary among classes
  - Notice `Naive Bayes` and `GDA` are badlhy hit by Imbalance. Class priors `p(y)` will get mor epoor as imabalnce imcreases
  - Srandard Logistic Regression: The Majority class will mostly guide the classifier. notice if there are 100 examples of (+) class and     just 5 of (-) class, the (+) class will `pull` the boundary towards itself to increase Average log likelihood
  - SVM: slightly better: it just looks at "nearest points". Recall: there is atleast 1 SV from each class. Can also weigh geometric margin for (+) and (-) differnetly and tune the model
  
* Compare OvO and OvM
  - One-vs-Rest (OvR) or One-vs-All (OvA):
  - For N classes, and this can be computationally efficient.
  - Inherent Probabilities: OvR classifiers can give you probability estimates directly, which is useful in applications like ranking or     risk assessment.
  - less efficient use of data: for example, if class1 and class2 are "sharply" differentiable, we aren't taking that
    benefit and "diluting" that. If the classes are already not well separated, it's even a bigger issue
  - We get direct probability values. But are they usefu?

  - One-vs-One (OvO):
  - Data Efficiency: OvO can be more data-efficient than OvR because each classifier is trained on a smaller subset of the data. This can be beneficial when data is limited.
  - Consistency: The output scores are directly comparable. Each classifier determines the "winner" for one pair of classes, and the class with the most "votes" wins.
  - Scalability: N(N-1)/2 classifiers needed for N classes
  - Inconsistent Probabilities: OvO classifiers do not give probability estimates directly, so you need to combine the votes or outputs in some way to estimate class probabilities.
 
* Domain Adaptation:
  - In out Naive Bayes model most if the data is general say "Positive" while in covid data, data is not Neutral "Negative", hence class priors are badly affected. This might be one reason for low accuracies in this case
  - Domain adaptation is necessary when the data distribution in the source domain does not fully align with the data distribution in the target domain. This misalignment can lead to poor model performance on the target domain.
* Why bigram? Words like Red Wine, North America, United States appear together. Thye loose meaning iof they are spit. Simlarly, if the phrase "not good" are split, they give the "oppositie" sentiment

* Why K-fold Cross validation is used in general?
  - When data is samll, so that we don't "waste" data for the validation set
* Soft Margin Classifier in linit of $C \to 0$ and $C \to \infty$
  - $C \to \infty$: there is infinite penalty for being on the wrong side of the Margin. Since we're trying to minimise the overall          cost, i.e. $||w||^2$ + penalty, this means, a feasible solution
    must have all points exactly on the correct side of the classifier, that is the data must be linearly separable
    for a solution to exits. This is the Hard SVM classifier.
  - $C \to 0$: there is no penaty for breaking the margin. We just need to minimise $||w||^2$. In this case the solution
    will just be $w = 0$ and $b$ can take any values and $\epsilon_i$'s can be set to values to make the constraints satisfy, without any penalty. 

* How does $||w||^2$ appear in the hard SVM classifier objective?
  - We try to maximuse the minimum gemometric margin. Singe geometric margin is invariant to rescaling, we set another contraint,
    that the minimum dunctional margin to 1, hence the minimum geometic margin becomes $1/||w||$, minimising which is equivalent to
    maximising ||w||
* Stemming vs Lematisation
* Confision Matrix needs to have the colour density 
* Can you use Gradient Descent to solve the Soft Margin SVM?
  - In the standard form: No. There are constraints and Gradient Descent won't respect that
    $$min_{ξ,w,b} \frac{1}{2}  \lVert w \rVert^2 + C \sum_{i=1}^m ξ_i$$
    under:
    $$y^{(i)}(w^Tx^{(i)} + b) \geq 1 - ξ_i , i = 1 , 2 ...m$$
    $$ξ_i \geq 0, i = 0 , 1...m$$
  - From this we get  $ξ_i \geq  1 - y^{(i)}(w^Tx^{(i)} + b)$
  - Which gives $ξ^{\*}_i =  max(0 , 1 - y^{(i)}(w^{\*T}x^{(i)} + b^{\*}))$
  - Above holds, as if $ξ^{\*}_i$ is any lesser, it violates the contraints, if it's
    any higher, we can prove by contradiction that is is suboptimal.
  
  - This leads us to the objective:
    $$min_{w,b} \frac{1}{2}  \lVert w \rVert^2 + C \sum_{i=1}^m max(0 , 1 - y^{(i)}(w^{\*T}x^{(i)} + b^{\*}))$$
    and $w$ and $b$ are now unconstrained.
  - $max$ is continuous, so this fuction is continuous everywhere
  - The second derivative may not exits at some points, but still note that this function is pievewise convex
  - So we use "sub gradient" toi solve it
* 
    


  
