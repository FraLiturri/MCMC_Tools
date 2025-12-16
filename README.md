# Tools for Markov Chain Monte Carlo analysis

Here are collected some useful tools for Markov Chain data analysis.
Currently have been implemented: 

- Bootstrap
- Jackknife
- Integrated autocorrelation time calculator

Both Bootstrap and Jackknife accept two keyword arguments: `primary function : list(callable)` and `function : callable`.The first one is the list of primary functions $g_{\alpha}(x)$ from which $F(x)$, passed through `function`, depends. 

Example: for $F(x) = \langle x ^4\rangle/\langle x^2 \rangle^2$, the inputs are `primary_functions = [lambda x: x**4, lambda x: x**2]` and lastly `function = lambda a,b : a/b**2`. In other words, `primary_funtions` contains all the functions whose means are computed, while `function` defines and registers the relation between these mean values. 
