include(joinpath(@__DIR__, "utilities.jl"))
include(joinpath(@__DIR__, "binarize.jl"))
include(joinpath(@__DIR__, "factor.jl"))

#=
Using the list of equations provided in the ODESystem an auxilliary system of
ODEs is formed by binarize'ing the underlying expressions (i.e., each expression is 
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

or 
              @parameters t σ ρ β
@variables x(t) y(t) z(t)
D = Differential(t)

eqs = [D(D(x)) ~ σ*(y-x),
       D(y) ~ x*(ρ-z)-y,
       D(z) ~ x*y - β*z]

@named sys = ODESystem(eqs)

@show sys.iv
@show sys.states
@show sys.eqs
@show sys.ps

# input
function rhs(dx,x,t,p)
    dx[1] = p[1]*x[1]^2 + p[1]
end

# 
function rhs(dx,x,t,p)
    v1 = x[1]^2
    v2 = p[1]*v1
    v3 = v2 + p[1]
    dx[1] = p[1]*x[1]^2 + p[1]
end

function rhs(dx,x,t,p)

    p1_lo = p[1]       # p[1] lo unpack parameter
    p1_hi = p[2]       # p[1] hi
    x1_lo = x[1]       # unpack state
    x1_hi = x[2]

    sqr_x1_I = Interval(x1_lo,x1_hi)^2                          # rule for v[1] = x[1]^2
    v1_lo = sqr_x1_I.lo
    v1_hi = sqr_x1_I.hi

    mul_x1p1_I = Interval(v1_lo,v1_hi)*Interval(p1_lo,p1_hi)    # rule for v[2] = p[1]*v[1]
    v2_lo = mul_x1p1_I.lo
    v2_hi = mul_x1p1_I.hi

    plus_v2p1_I = Interval(v2_lo,v2_hi) - Interval(p1_lo,p1_hi) # rule for v[3] = p[1]*v[1]
    v3_lo = plus_v2p1_I.lo
    v3_hi = plus_v2p1_I.hi

    dx[1] = v3_lo      # dx[1] lo
    dx[2] = v3_hi      # dx[1] hi

    nothing
end


function rhs(dx,x,t,p)

    p1_lo = p[1]       # p[1] lo unpack parameter
    p1_hi = p[2]       # p[1] hi
    x1_lo = x[1]       # unpack state
    x1_hi = x[2]

    v1_lo = sqr_lo(x1_lo, x1_hi)
    v1_hi = sqr_hi(x1_lo, x1_hi)

    v2_lo = mul_lo(v1_lo, v1_hi, p1_lo, p1_hi)
    v2_hi = mul_hi(v1_lo, v1_hi, p1_lo, p1_hi)

    v3_lo = # SO ON...
    v3_hi = 

    dx[1] = v3_lo      # dx[1] lo
    dx[2] = v3_hi      # dx[1] hi
    
    nothing
end
        
```
        
=#
function apply_transform(transform::T, prob::ODESystem) where T<:AbstractTransform

    # Factorize all model equations to generate a new set of equations

    genparam(get_name(prob.iv.val))

    equations = Equation[]
    for eqn in prob.eqs
        current = length(equations)
        factor!(eqn.rhs, eqs=equations)
        if length(equations) > current
            push!(equations, Equation(eqn.lhs, equations[end].rhs))
            deleteat!(equations, length(equations)-1)
        else
            index = findall(x -> isequal(x.rhs, eqn.rhs), equations)
            push!(equations, Equation(eqn.lhs, equations[index[1]].lhs))
        end
    end

    # Apply transform rules to the factored equations to make the final equation set
    new_equations = Equation[]
    for a in equations
        zn = var_names(transform, zstr(a))
        xn = var_names(transform, xstr(a))
        if isone(arity(a)) 
            targs = (transform, op(a), zn..., xn...)
         else
            targs = (transform, op(a), zn..., xn..., var_names(transform, ystr(a))...)
        end
        new = transform_rule(targs...)
        for i in new
            push!(new_equations, i)
        end
    end

    # println("")
    # println("Old equations:")
    # for i in prob.eqs
    #     println(i)
    # end
    # println("")
    # println("New equations:")
    # for i in new_equations
    #     println(i)
    # end
    # println("")

    # Copy model start points to the newly transformed variables
    var_defaults, param_defaults = translate_initial_conditions(transform, prob, new_equations)
    # for i in param_defaults
    #     print(i)
    #     print("\n")
    # end
    # for i in var_defaults
    #     print(i)
    #     print("\n")
    # end
    

    # Use the transformed equations and new start points to generate a new ODE system
    # @named new_sys = ODESystem(new_equations, default_u0=var_defaults, default_p=param_defaults)

    @named new_sys = ODESystem(new_equations, defaults=merge(var_defaults, param_defaults))

    #Extract RHS and evaluate at a point, make that a script that tests any functions.
    # RHS this function, reference the correct thing, go

    return new_sys
end

# Separate case for applying transform to only a set of equations
function apply_transform(transform::T, eqn_vector::Vector{Equation}) where T<:AbstractTransform

    # Factorize all model equations to generate a new set of equations
    equations = Equation[]
    for eqn in eqn_vector
        current = length(equations)
        factor!(eqn.rhs, eqs=equations)
        if length(equations) > current
            push!(equations, Equation(eqn.lhs, equations[end].rhs))
            deleteat!(equations, length(equations)-1)
        else
            index = findall(x -> isequal(x.rhs, eqn.rhs), equations)
            push!(equations, Equation(eqn.lhs, equations[index[1]].lhs))
        end
    end

    # Apply transform rules to the factored equations to make the final equation set
    new_equations = Equation[]
    for a in equations
        zn = var_names(transform, zstr(a))
        xn = var_names(transform, xstr(a))
        if isone(arity(a)) 
            targs = (transform, op(a), zn..., xn...)
         else
            targs = (transform, op(a), zn..., xn..., var_names(transform, ystr(a))...)
        end
        new = transform_rule(targs...)
        for i in new
            push!(new_equations, i)
        end
    end

    return new_equations
end
function apply_transform(transform::T, eqn::Equation) where T<:AbstractTransform

    # Factorize the equations to generate a new set of equations
    equations = Equation[]
    current = 0
    factor!(eqn.rhs, eqs=equations)
    if length(equations) > current
        push!(equations, Equation(eqn.lhs, equations[end].rhs))
        deleteat!(equations, length(equations)-1)
    else
        index = findall(x -> isequal(x.rhs, eqn.rhs), equations)
        push!(equations, Equation(eqn.lhs, equations[index[1]].lhs))
    end

    # Apply transform rules to the factored equations to make the final equation set
    new_equations = Equation[]
    for a in equations
        zn = var_names(transform, zstr(a))
        xn = var_names(transform, xstr(a))
        if isone(arity(a)) 
            targs = (transform, op(a), zn..., xn...)
         else
            targs = (transform, op(a), zn..., xn..., var_names(transform, ystr(a))...)
        end
        new = transform_rule(targs...)
        for i in new
            push!(new_equations, i)
        end
    end

    return new_equations
end