#Fitting Methods and Scatter

Construct two 100-element “data sets” x and y such that x ranges from 1-10 and y ranges from 20-40.  Note that x and y should vary smoothly, with no randomness. If you plot y vs. x, you see the “true” relation with no measurement errors or biases.

Now add random Gaussian scatter to y with a sigma of 1. Also choose ~10 elements of y to give extra “systematic” errors of 2-3 by hand (hint – systematic errors all go in one direction, unlike random errors).  Plot y vs. x. Fit the data using forward, inverse, and bisector fits and overplot the fits. The bisector slope is [β1+β2-1 + √((1+β12)(1+β22))] / (β1 + β2) and you may calculate its intercept by noting that it passes through the intersection point of the other two fits. Label the fits and comment on which one appears most correct. For each fit, compute the rms scatter in the y-direction. In this case, the lowest rms scatter corresponds to the most correct fit – why?

Now, add Gaussian scatter to x with a sigma of 3 and repeat your fits. Which type of fit appears most correct now? Consider your “gut feeling” as well as the original true relation. Why might these not agree? Recompute the rms scatter in the y-direction. Why does the lowest rms scatter not correspond to the best fit anymore? Can you see another way of computing the rms scatter by which the best fit would in fact correspond to the lowest scatter?

Finally, add a selection bias on x, such that x cannot be detected below 3. Repeat your fits and again discuss which fit appears most correct vs. is actually most correct.

All of the above assumed that the goal was to measure the true, underlying relationship between x and y.  What if your goal were to find the best predictive relation between the two, in order to predict y with greatest accuracy for a given x. How would the optimal choice of fit type change in this case?
