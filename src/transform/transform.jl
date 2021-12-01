include(joinpath(@__DIR__, "utilities.jl"))
include(joinpath(@__DIR__, "binarize.jl"))
include(joinpath(@__DIR__, "factor.jl"))

#=
Using the list of equations provided in the ODESystem an auxilliary system of
ODEs is formed by binarize the underlying expressions (i.e. each expression is 
written either as a variable or a function with arity one or two), then a variable
is added for each factor present in the ODE, each factor is then replaced with an expression
for constructing intervals/relaxations, then a new ODESystem is created.


apply_transform(IntervalTransform(), odes) should create a 2*nx dimension system of differential equations with rhs equal to f(du, u, p ,t)
from the original with rhs f(dx, x, p, t) which when solved furnishes interval bound of x(p,t).

apply_transform(McCormickIntervalTransform(), odes) should create a 4*nx dimension system of differential equations with rhs equal to f(du, u, p ,t)
from the original with rhs f(dx, x, p, t) which when solved furnishes relaxations of x(p,t)
=#
function apply_transform(t::T, odes::ODESystem) where T<:AbstractTransform

    assigments = Assignment[]
    for eqn in equations(prob.odes)
        binarize!(eqn)
        factor!(eqn, assigments)
    end

    new_assignments = Assignment[]
    for a in assignments
        zn = var_names(t, zstr(a))
        xn = var_names(t, xstr(a))
        if isone(arity(a)) 
            targs = (t, op(a), zn..., xn...)
         else
            targs = (t, op(a), zn..., xn..., var_names(t, ystr(a))...)
        end
        add!(new_assignments, transform_rule(targs...))
    end

    # Form ODE system from new assignments
    # CSE - MTK.structural_simplify()

    return new_assignments
end