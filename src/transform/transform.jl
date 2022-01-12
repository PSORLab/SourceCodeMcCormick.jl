include(joinpath(@__DIR__, "utilities.jl"))
include(joinpath(@__DIR__, "binarize.jl"))
include(joinpath(@__DIR__, "factor.jl"))

#=
Using the list of equations provided in the ODESystem an auxilliary system of
ODEs is formed by binarize the underlying expressions (i.e. each expression is 
written either as a variable or a function with arity one or two), then a variable
is added for each factor present in the ODE, each factor is then replaced with an expression
for constructing intervals/relaxations, then a new ODESystem is created.


apply_transform(IntervalTransform(), odes) should create a 2*nx dimension system of differential
equations with rhs's equal to f(du, u, p ,t) from the original with rhs's f(dx, x, p, t) which 
when solved furnishes interval bounds of x(p,t) from an existing system of ODEs.

apply_transform(McCormickIntervalTransform(), odes) should create a 4*nx dimension system of 
differential equations with rhs's equal to f(du, u, p ,t) from the original with rhs's 
f(dx, x, p, t) which when solved furnishes relaxations of x(p,t)

```
# Sample ODESystem from Modeling Toolkit
using ModelingToolkit, OrdinaryDiffEq
@parameters t σ ρ β
@variables x(t) y(t) z(t)
D = Differential(t)

eqs = [D(D(x)) ~ σ*(y-x),
       D(y) ~ x*(ρ-z)-y,
       D(z) ~ x*y - β*z]

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

new_sys, new_u0, new_p = apply_transform(McCormickIntervalTransform(), sys, u0, p, args...)
# new_sys is the auxillary rhs odes that define the relaxations
# new_p is a vector of [p;pL;pU] defining the box and point of evaluation
# new_u0 is a function of p that evaluates to the initial condition

# creating an auxillary ODE problem
new_prob = ODEProblem(sys,u0,tspan,p,jac=true)
        
# evaluating a relaxation is then done by...       
        
# 1. setting the initial condition... pval, pL, pU are vectors of length p with pL <= p <= pU
pMC = MC{length(P),NS}(pval, pL, pU)
u0MC = u0(pMC)
        
# 2. remaking problem with desired p value
pnew = [p; pL; pU]
new_prob_p = remake(new_prob, p=pnew)
        
# 3. solve ODE
# 4. extract values

        
```
        
=#
function apply_transform(t::T, prob::ODESystem) where T<:AbstractTransform

    assignments = Assignment[]
    for eqn in prob.eqs
        # Flesh out the original RHS
        current = length(assignments)
        factor!(toexpr(eqn.rhs), assignments=assignments)

        # If new equations were added, stick on the original LHS to the last point 
        # in assignments by stealing its RHS from the last item and taking its place
        if length(assignments) > current
            push!(assignments, Assignment(toexpr(eqn.lhs), assignments[end].rhs))
            deleteat!(assignments, length(assignments)-1)
        end
    end

    # new_assignments = AssignmentPair[]
    new_assignments = Assignment[]
    for a in assignments
        zn = var_names(t, zstr(a)) #LHS
        xn = var_names(t, xstr(a)) #RHS(1)

        first = zn[1]
        second = zn[2]

        # Define the zn's as variables
        @variables $first $second
        if isone(arity(a)) 
            targs = (t, op(a), zn..., xn...)
         else
            targs = (t, op(a), zn..., xn..., var_names(t, ystr(a))...)
        end
        println("targs: $targs")

        # push!(new_assignments, transform_rule(targs...))
        new = transform_rule(targs...)
        push!(new_assignments, new.l)
        push!(new_assignments, new.u)
    end

    @named new_sys = ODESystem(new_eqs)
    # Form ODE system from new assignments
    # CSE - MTK.structural_simplify()

    # Figure out a way to give the new ODE system the proper parameters, variables, etc.
    return the_new_ODE_System
end
