using ModelingToolkit

@variables t x1(t) x2(t) x3(t) x4(t)
@parameters p1 p2 p3 p4 p5
D = Differential(t)

@named testcase = ODESystem(
    [D(x1) ~ -(p1+p2+p3)*x1 + p5*x4,
     D(x2) ~ (p2*x1) + -(p4*x2),
     D(x3) ~ (p3*x1 + p4*x2),
     D(x4) ~ p1*x1 - p5*x4])

using DifferentialEquations: solve
using Plots: plot
prob = ODEProblem(structural_simplify(testcase), [x1 => 1, x2 => 0, x3 => 0, x4 => 0], 
    (0.0, 120.0), [p1 => 0.02, p2 => 0.03, p3 => 0.04, p4 => 0.05, p5 => 0.06])
sol = solve(prob)
plot(sol, vars=[x1, x2, x3, x4])

new = toexpr(testcase.eqs[1].rhs)
binarize!(new)
factor!(new)

@named test2 = ODESystem([D(x1) ~ (p1+p2+p3),
                          D(x2) ~ p1*x1,
                          D(x3) ~ x1])
t = toexpr(test2.eqs[1].rhs)
factor!(toexpr(test2.eqs[1].rhs))

@parameters t1


apply_transform(IntervalTransform(), testcase)


# THIS IS MY TEST CASE, THESE SIX LINES
using ModelingToolkit, OrdinaryDiffEq
@parameters t
@variables x(t) y(t)
D = Differential(t)
@named test = ODESystem([D(x) ~ exp(x+y^2)])
apply_transform(IntervalTransform(), test)


# THIS TEST DOES NOT WORK, BECAUSE THE VECTOR-VALUED PARAMETERS
# AND VARIABLES MEAN I NEED TO PRE-EMPTIVELY MAKE/USE OVER/UNDERESTIMATES
# (This is possible but would need a little bit of a rewrite/change. and
# some thinking about how to handle cases like that)
using ModelingToolkit, OrdinaryDiffEq
@parameters t
@variables x[1:2,1:3,1:4](t)
D = Differential(t)
@named test2 = ODESystem([D(x[1]) ~ x[1]*x[2]])
apply_transform(IntervalTransform(), test2)


# Trying out getting the thing in a variable form
a = gensym(:aux)
b = Symbol(string(a)[3:5] * string(a)[7:end])
c = genvar(b)[1]



@named test3 = ODESystem([D(x1) ~ 5,
                          D(x2) ~ x1*p1])
apply_transform(IntervalTransform(), test3)



a = gensym(:test)
# b = :(Symbolics._parse_vars(:variables, Real, $a))
# eval(b)
c = Symbolics._parse_vars(:variables, Real, a)


using ModelingToolkit, OrdinaryDiffEq
a = gensym(:var)
@parameters t σ ρ β
@variables x(t) y(t) z(t) $a(t)
D = Differential(t)

eqs = [D(D(x)) ~ σ*(y-x),
       D(y) ~ x*(ρ-z)-y,
       D(z) ~ x*y - β*z]

eqs = [D(x) ~ σ+x]

@named sys = ODESystem(eqs)
sys = ode_order_lowering(sys)

u0 = [D(x) => 2.0,
      x => 1.0,
      y => 0.0,
      z => 0.0]

p  = [σ => 28.0,
      ρ => 10.0,
      β => 8/3]

tspan = (0.0,100.0)
prob = ODEProblem(sys,u0,tspan,p,jac=true)

println(typeof(eqs))
println(length(eqs))
apply_transform(IntervalTransform(), sys)


## New test!
using ModelingToolkit, OrdinaryDiffEq
@parameters t p
@variables x(t)
eqs = [D(x) ~ p[1]*x[1]^2 + p[1]]

# Try to do what @variables does, but with a symbol of choice
Symbolics._parse_vars(:variables, Real, xs)
# where xs is xs... is the names of the symbols to make into variables

a = gensym(:new)
b = Symbol(string(a)[3:5] * string(a)[7:end])

genvar(b)
genvar(b, [:t, :m])


a = gensym(:var)
println("The new variable a is $a")

for i = 1:1000
    gensym(:tes)
end


# function Base.iterate(a::Symbol)
#     return (a, nothing)
# end
# function Base.iterate(a::Symbol, ::Nothing)
#     return (a, nothing)
# end
Base.iterate(a::Symbol) = (a, nothing)
Base.iterate(a::Symbol, state) = nothing


# Original system from HDO Blackbox
# dx[1] = -(p[1]+p[2]+p[3])*x[1] #Guaiacol
# dx[2] = (p[2]*x[1]) + -(p[4]*x[2]) #Methoxycyclohexanone
# dx[3] = (p[3]*x[1] + p[4]*x[2]) #Cyclohexanol
# dx[4] = p[1]*x[1] #Fouled amount

# @variables t x(t) RHS(t)  # independent and dependent variables
# @parameters τ       # parameters
# D = Differential(t) # define an operator for the differentiation w.r.t. time

# # your first ODE, consisting of a single equation, indicated by ~
# @named fol_separate = ODESystem([ RHS  ~ (1 - x)/τ,
#                                   D(x) ~ RHS ])


# using DifferentialEquations: solve
# using Plots: plot

# prob = ODEProblem(structural_simplify(fol_separate), [x => 0.0], (0.0,10.0), [τ => 3.0])
# sol = solve(prob)
# plot(sol, vars=[x,RHS])


xs = [1,2,3,4,5]
function test!(vals::Vector{Int64})
    p = collect(1:length(vals))
    deleteat!(p, 3)
    push!(p, 3)
    vals[:] = vals[p]
end