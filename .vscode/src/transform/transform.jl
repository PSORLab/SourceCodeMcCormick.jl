include(joinpath(@__DIR__, "utilities.jl"))
include(joinpath(@__DIR__, "binarize.jl"))
include(joinpath(@__DIR__, "factor.jl"))

#=
Using the list of equations provided in the ODESystem an auxilliary system of
ODEs is formed by binarize the underlying expressions (i.e. each expression is 
written either as a variable or a function with arity one or two), then a variable
is added for each factor present in the ODE, each factor is then replaced with an expression
for constructing intervals/relaxations, then a new ODESystem is created.
=#
function apply_transform(t::T, odes::ODESystem) where T<:AbstractOverload

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