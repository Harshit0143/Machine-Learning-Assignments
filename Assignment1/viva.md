* Among Gradient Descent and Newton Raphson whaich would you choose to optimise?
  - Notice that the matrix inversion at each stage in Newton Raphson takes $O(n^3)$ time.
  - So if the number of features is large, even 1 interation gets very expensive
  - Also Newton's method needs a stronger requiremennt: THe second gradient of $J(\theta)$ needs to exist
  - Even if it exists, we might face the problem of "Is H invertible?"
  - I was asked to ignore the above. Now can you pose somethog?
  - Gradient descent is better as you can "tweak" the learning rate. Consider a fuciton that is not convex and
   your start point is close to a local minima, Newton Raphson is bound to converge on that miniam. For Gradient Desent, you
   still have the option of tweaking the learning rate. (Vague. I don't agree yet)
   
* In Logistic Regression why can't you take sum squared difference as the
  $$\frac{1}{2m} \sum_{i = 1}^m h_{\theta}(x^{(i)}) - y^{(i)}$$
  where $$h_{\theta} (x) = sigmoid(\theta^T x)$$
  - My first response was the way in which they penalise, $log$ will penalise the "misclassified" values very strongle
  - Notice the derivative of `log(x)` close to right side of 0. Why is this relevant? In average LL maximisation, the maximisation objective, is
    $$\frac{1}{m}\sum_{i = 1}^m y^{(i)}log(\hat{y}^{(i)}) + (1 - y^{(i)}) log(1 - \hat{y}^{(i)})$$
  - Say $y^{(i)} = 1$ then the corresponding term is $log(\hat{y}^{(i)})$
  - Notice as $\hat{y}^{(i)}$ gets, closer to $0$, i.e. more and more wrong, the penalty increases and "very strongly"
  - While a quadratic function, quanlitatively doesn't offer such a penalty for "misclassification"
  - The correct answer: use of squared sum error in Regression and LL error in Logistic regression are backed by their
    corresponding Probabilistic Interpretations and we can't pull out an error out of the blue and claim it should be used

* In probabilistic Interpreation of Linear Regression we take the $\epsilon_i$'s to be iid. What if we take then to be independent, mean zero but they can have
  different variances?
  - Using the Probabilistic Interpretation we derive:
   $$J(\theta) = \frac{1}{2m} \sum_{i = 1}^m \frac{(\theta^Tx^{(i)} - y^{(i)})^2}{\sigma_i}$$ 
  - Intuitively we say, there's still a quadratic pealty for $\hat{y}^{(i)}$ deviating from $y^{(i)}$ but each point is given a diffent weight
  - $i$'s corresponding to higher $\sigma_i$ have lower "penalty", intuitively as a high variance means we already $\sigma_i$ to be
    farther away from the mean $0$ 

* Is convergence in $J_{\theta}$ or convergence in $||\theta||$ better criteria for convergnece?
  - Just a weak answer stated: convergence in $||\theta||$ 
  - Back it as: convergence in $||\theta||$ implies convergence in $J_{\theta}$
  - Differentiate $J_{\theta}$ to prove above result
  - Further recall: the case of perfectly separable data in Logistic Regression. $||\theta||$ doesn't converge. While $J_{\theta}$ does
  - By keeping track of  $||\theta||$ you'd be able to figure out this "bug" in the above case
* Is GDA a subset of Logistic or the Reverse?
  - The language is absurd. The answer is clear thougH. Setting $\Sigma_0 = \Sigma_1$ in GDA and solving for the boundary
    $p(y = 1 | x)$ reduces to the logistic form of sigmoid.
* Show that Naive Bayes, in case of Binary, where each feature vector has size $|V|$ each entry, $0$ or $1$ denoting if the $i^{th}$
  word from Vocabulary is present in the email or not. This also reduces to Logistic
* MLE is a subset of MAP or MAP is a subset of MLE?
  - MAP assumes a priori distribution on $\theta$
  - SSetting $\theta$ = Uniform, it results in the MLE expression
 
  
    <img width="571" alt="Screenshot 2023-10-18 at 6 16 36 PM" src="https://github.com/Harshit0143/Machine-Learning-Assignments/assets/97736991/d9ae0c3e-10a6-456f-b28c-6155b79b0600">
    
* Explain why there's less "wobbling" in the start (far from optimum) and increases on going closer to the optimum?
  - Firstly note that there are $r$ different optima is if split the batch into $r$ pieces and iterate over them in a robin hood fashion (no shuffling done except from when the start)
  - Intuitively, the these optima are close to each other in position $(\theta_0, \theta_1,\theta_2)$. When we start far from the optima, notice by drawing arrows, from currect position in direction of each oprima, these arrows are closely aligned. Hence, in successive steps either batch "guides" $\theta$ in more or less the same direction
  - As you get closer, notice, the angle between these arrows increases, each sussessive batch guides in a diffeennt direction so wobbling increases

