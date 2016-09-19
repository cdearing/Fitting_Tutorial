Zombie Activity.

### zombies 1

For this activity you will write your own code, but remember that you can take portions of previous code to make the process faster. Use the python quick reference card (http://user.physics.unc.edu/~sheila/PythonQuickReference.pdf) and codes from previous tutorials to help you write the code for this activity. Please feel free to work together on this part. My solutions are provided in `zombies1.py.sln`.

A virus has gotten out that is turning humans into zombies. You have been recording the % of zombies ever since the outbreak (~14 days ago). However the power has gone out for the past four days and your generator just kicked in allowing you to analyze the data and determine when there will be no humans left (`% humans = [1- % zombies] = 0`). Your data are in `percentzombie.txt` where `time=0` is the present day (`time = -14` is 14 days ago when you started taking data).

1. Read in your data and plot it as % human vs. time as blue stars. The uncertainties on both time and % zombie are unknown.

2. Evaluate the MLE slope & y-intercept and overplot the best fit line in green. What does the y-intercept value mean in the context of this situation? Are you a zombie?

3. In the above step you have fit the data minimizing residuals in the y-direction (% human). How could you use `np.polyfit` to fit the data minimizing residuals in the x-direction (time)? Keep in mind that you can rewrite a line `y=a*x + b` as `x =(1/a)*y – (b/a)`. Over plot this fit in red – how does the y-intercept value change? Does this change your conclusion as to whether you are a zombie? In which variable should you minimize residuals to get the most accurate prediction of total zombification?

4. Now assume your uncertainty on each % zombie measurements is ~3%. In a new plotting window, plot the residuals in % human from the linear fit from part b as green stars (residuals can be computed for either % human or time, but use % human since we have an estimated error for each data point of about 3%). Evaluate the reduced χ<sup>2</sup> for your data using residuals in % human as that is the measurement for which we have an uncertainty estimate (refer to the Correlations & Hypothesis Testing Tutorial). Is your model a good fit to the data? If 3% is an over- or under-estimate of the errors, how will this affect the reduced χ<sup>2</sup>? Often times we think the R value from the Pearson correlation test tells us how good the fit is, but a reduced χ<sup>2</sup>actually gives a better estimate of how good your model is, not just whether the correlation is strong.

5. What happens when you increase the order of the fit (% humans vs. time)? Overplot the higher order fits on figure 1. What happens to the residuals if you increase the order of the fit (see **np.polyfit**, and optionally **np.polyval**)? Overplot the new residuals in time compared to the residuals from the linear fit from part b on figure 2.

6. Calculate the reduced χ<sup>2</sup>for these higher order fits – do they yield as good a fit to the data as the linear fit?