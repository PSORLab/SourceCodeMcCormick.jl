include(joinpath(@__DIR__, "utilities.jl"))
include(joinpath(@__DIR__, "binarize.jl"))
include(joinpath(@__DIR__, "factor.jl"))


function apply_transform(transform::T, prob::ODESystem; constants::Vector{Num}=Num[]) where T<:AbstractTransform

    # Factorize all model equations to generate a new set of equations

    genparam(get_name(prob.iv.val))

    equations = Equation[]
    for eqn in prob.eqs
        current = length(equations)
        factor(eqn.rhs, eqs=equations)
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
        if string(xstr(a)) in string.(constants)
            xn = (xstr(a), xstr(a), xstr(a), xstr(a))
        else
            xn = var_names(transform, xstr(a))
        end
        if isone(arity(a)) 
            targs = (transform, op(a), zn..., xn...)
        else
            if string(ystr(a)) in string.(constants)
                yn = (ystr(a), ystr(a), ystr(a), ystr(a))
            else
                yn = var_names(transform, ystr(a))
            end
            targs = (transform, op(a), zn..., xn..., yn...)
        end
        new = transform_rule(targs...)
        for i in new
            push!(new_equations, i)
        end
    end

    # Copy model start points to the newly transformed variables
    var_defaults, param_defaults = translate_initial_conditions(transform, prob, new_equations)

    # Use the transformed equations and new start points to generate a new ODE system
    @named new_sys = ODESystem(new_equations, defaults=merge(var_defaults, param_defaults))

    return new_sys
end

function apply_transform(transform::T, eqn_vector::Vector{Equation}; constants::Vector{Num}=Num[]) where T<:AbstractTransform

    # Factorize all model equations to generate a new set of equations
    equations = Equation[]
    for eqn in eqn_vector
        current = length(equations)
        factor(eqn.rhs, eqs=equations)
        if length(equations) > current
            push!(equations, Equation(eqn.lhs, equations[end].rhs))
            deleteat!(equations, length(equations)-1)
        else
            index = findall(x -> isequal(x.rhs, eqn.rhs), equations)
            push!(equations, Equation(eqn.lhs, equations[index[1]].lhs))
        end
    end

    # Apply transform rules to the factored equations to make the final equation set
    # NOTE: Unlike the ordering of [cv,cc,L,U] used in other parts of the code, the
    #       ordering used by transformations is [L,U,cv,cc]. This is because convex
    #       and concave relaxations may occasionally need previously calculated lower
    #       and upper bounds, and it is more convenient to list the interval extension
    #       first so that the substitution step can proceed without backtracking.
    new_equations = Equation[]
    for a in equations
        zn = var_names(transform, zstr(a))
        if string(xstr(a)) in string.(constants)
            xn = (xstr(a), xstr(a), xstr(a), xstr(a))
        else
            xn = var_names(transform, xstr(a))
        end
        if isone(arity(a)) 
            targs = (transform, op(a), zn..., xn...)
        else
            if string(ystr(a)) in string.(constants)
                yn = (ystr(a), ystr(a), ystr(a), ystr(a))
            else
                yn = var_names(transform, ystr(a))
            end
            targs = (transform, op(a), zn..., xn..., yn...)
        end
        new = transform_rule(targs...)
        for i in new
            push!(new_equations, i)
        end
    end

    return new_equations
end

function apply_transform(transform::T, eqn::Equation; constants::Vector{Num}=Num[]) where T<:AbstractTransform

    # Factorize the equations to generate a new set of equations
    equations = Equation[]
    factor(eqn.rhs, eqs=equations)
    if length(equations) > 0
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
        if string(xstr(a)) in string.(constants)
            xn = (xstr(a), xstr(a), xstr(a), xstr(a))
        else
            xn = var_names(transform, xstr(a))
        end
        if isone(arity(a)) 
            targs = (transform, op(a), zn..., xn...)
        else
            if string(ystr(a)) in string.(constants)
                yn = (ystr(a), ystr(a), ystr(a), ystr(a))
            else
                yn = var_names(transform, ystr(a))
            end
            targs = (transform, op(a), zn..., xn..., yn...)
        end
        new = transform_rule(targs...)
        for i in new
            push!(new_equations, i)
        end
    end

    return new_equations
end

function apply_transform(transform::T, num::Num; constants::Vector{Num}=Num[]) where T<:AbstractTransform

    # Factorize the equations to generate a new set of equations
    @variables result
    eqn = result ~ num
    equations = Equation[]
    factor(eqn.rhs, eqs=equations)
    if length(equations) > 0
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
        if string(xstr(a)) in string.(constants)
            xn = (xstr(a), xstr(a), xstr(a), xstr(a))
        else
            xn = var_names(transform, xstr(a))
        end
        if isone(arity(a)) 
            targs = (transform, op(a), zn..., xn...)
        else
            if string(ystr(a)) in string.(constants)
                yn = (ystr(a), ystr(a), ystr(a), ystr(a))
            else
                yn = var_names(transform, ystr(a))
            end
            targs = (transform, op(a), zn..., xn..., yn...)
        end
        new = transform_rule(targs...)
        for i in new
            push!(new_equations, i)
        end
    end

    return new_equations
end