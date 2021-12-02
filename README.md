# SourceCodeMcCormick.jl
Experimental Approach to McCormick Relaxation Source-Code Transformation in Differential Inequalities

Mainly exists to avoid need for method overloading and use of structures which are not sufficiently generically typed to when computing relaxations via differential inequality.

Main Features: 
- Transform factorable function `y = f(x<:Real)` into `y = f(xcv<:Real, xcc<:Real, xL<:Real, xU<:Real)` outputs the tuple `(ycv, ycc, yL, yU)` instead and computes the convex relaxation (ycv), concave relaxation (ycc), lower bound (yL), and upper bound of `f(x)` with `xL < xc < x < xcc < xU`.
- Transform the factorable function `f!(y::Vector{<:Real}, x::Vector{<:Real})` which maps `x` to `y` similarly.  
- Transform a list of equations from Modeling toolkit `eqn` that are a parametric ODEs with defined by rhs function `f!(dy,y,p,t)` to a list of equations in Modeling toolkit `eqn_new` similarly.
