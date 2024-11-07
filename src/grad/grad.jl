
# Can remove the import statement once this is fully incorporated into SCMC
# import SourceCodeMcCormick: xstr, ystr, zstr, var_names, arity, op, transform_rule


"""
    grad(::Num; force::Bool)
    grad(::Num, ::Vector{Num}; base_level::Bool, force::Bool)

Given a symbolic expression, return vectors of expressions
representing the subgradients of the convex and concave
relaxations. Inputs are the expression that subgradients are
requested for and, optionally, the dimensions that are needed.
If no `Vector{Num}` is given, subgradients will be produced
with respect to all variables found in the expression.

By default, `grad` will assume that the input is a subexpression
which is part of a larger expression that subgradients are desired
for. Therefore, values for the gradients will be required as inputs
to the created functions. Alternatively, if `base_level` is set
to `true`, gradients will be constructed of `0`s and a `1` based
on the order of variables given in the `Vector{Num}`.

This function includes a check to make sure excessively large
substitutions are not being made, which can stall Julia for 
potentially long time period. If large substitutions are detected,
`grad` will back out of calculations and warn the user. This
functionality can be suppressed by setting `force=true`.

# Example

```
cvgrad, ccgrad = grad(x*y, [x,y,z], base_level=true);
```

Here, subgradients are requested for the expression `x*y`. The user
has indicated that the full expression being considered is 3-dimensional,
with dimensions `[x,y,z]`, so the resulting gradient expressions will
also be 3-dimensional. E.g., `cvgrad` will be a 3-element Vector{Num}
with elements `cvgrad[1]` being the x-component of the subgradient of 
the convex relaxation of `x*y`, `cvgrad[2]` being the y-component, and
`cvgrad[3]` being the z-component. Because `base_level` has been set to
`true`, the gradients of `x`, `y`, and `z` are internally set to
`[1,0,0]`, `[0,1,0]`, and `[0,0,1]`, respectively, prior to creating
the subgradient expressions for the convex relaxation of `x*y`. Note 
that all of the above also applies to `ccgrad`, which contains the 
subgradients of the concave relaxation of the input expression. 
    
If `base_level` were not set to `true` (and instead retained its default
value of `false`), the resulting subgradient expressions would be functions
of `[dx/dx, dx/dy, dx/dz]`, `[dy/dx, dy/dy, dy/dz]`, and `[dz/dx, dz/dy, dz/dz]`,
respectively. This may be important if expressions are broken into subexpressions
that are individually fed to `grad`. E.g.,:

```
cvgrad, ccgrad = grad(a*b, [x,y,z])
```

In this case, `a` and `b` may be composite, intermediate terms, which contain
the base-level variables `x`, `y`, and `z`. The resulting expressions from this
call of `grad` would then require inputs of the McCormick tuples of `a` and `b`, 
as well as values for `[da/dx, da/dy, da/dz]` and `[db/dx, db/dy, db/dz]`. 

If `grad` is called with only the first argument, `base_level` will be
assumed to be `true` (i.e., it is assumed that if the user is writing an
expression and wants only the variables present in the expression to be
used in the creation of subgradient expressions, the expression is likely
made only of base-level variables).
"""
grad(num::Num; force::Bool=false, expand::Bool=false, constants::Vector{Num}=Num[]) = grad(num, pull_vars(num), force=force, expand=expand, constants=constants)
function grad(num::Num, varlist_in::Vector{Num}; force::Bool=false, expand::Bool=false, constants::Vector{Num}=Num[])

    # Create a new varlist to keep track of which gradients are being tracked
    orig_len = length(varlist_in)
    varlist_in_string = string.(get_name.(varlist_in))
    varlist = copy(varlist_in_string)

    # Add in variables that weren't included in the varlist
    all_vars = string.(get_name.(pull_vars(num)))
    for var in all_vars
        if !(var in varlist)
            push!(varlist, var)
        end
    end
    base_vars = copy(varlist)
    base_len = length(varlist)

    # Factorize the equation to generate a new set of equations
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

    # Add in auxiliary variables that appeared during factorization
    varlist = [varlist; string.(get_name.(getfield.(equations, :lhs)))]
    
    # Now we need to create vectors of type Num, of size (orig_len, length(varlist))
    cv_gradlist = Num.(zeros(orig_len, length(varlist)))
    cc_gradlist = Num.(zeros(orig_len, length(varlist)))
    for i in 1:base_len
        if !expand && (base_vars[i] in varlist_in_string)
            # If the variable is in the input list, we know it's a base-level variable and we know its gradient
            cv_gradlist[i,i] = Num(1.0)
            cc_gradlist[i,i] = Num(1.0)
        elseif string(base_vars[i]) in string.(constants)
            # If the variable is a constant input, the gradients are always zero
            nothing
        else
            for j = 1:orig_len
                # If the variable isn't in the list, assume it's some function of the base-level variables
                cv_gradlist[j,i] = genvar(Symbol("∂"*varlist[i]*"∂"*varlist_in_string[j]*"_cv"))
                cc_gradlist[j,i] = genvar(Symbol("∂"*varlist[i]*"∂"*varlist_in_string[j]*"_cc"))
            end
        end
    end

    # Apply transform rules to the factored equations to make the final equation set.
    # Within each equation, we also want to update the gradlist.
    new_equations = Equation[]
    for a in equations
        zn = var_names(McCormickIntervalTransform(), zstr(a))
        if string(xstr(a)) in string.(constants)
            xn = (xstr(a), xstr(a), xstr(a), xstr(a))
        else
            xn = var_names(McCormickIntervalTransform(), xstr(a))
        end
        if isone(arity(a)) 
            targs = (McCormickIntervalTransform(), op(a), zn..., xn...)
        else
            if string(ystr(a)) in string.(constants)
                yn = (ystr(a), ystr(a), ystr(a), ystr(a))
            else
                yn = var_names(McCormickIntervalTransform(), ystr(a))
            end
            targs = (McCormickIntervalTransform(), op(a), zn..., xn..., yn...)
        end
        new = transform_rule(targs...)
        for i in new
            push!(new_equations, i)
        end

        # Apply the appropriate transform rule to propagate the subgradients
        # in cv_gradlist and cc_gradlist
        # grad_transform!(McCormickIntervalTransform(), op(a), new[1].rhs, new[2].rhs, new[3].rhs, new[4].rhs, targs[7:end]..., varlist, cv_gradlist, cc_gradlist)
        grad_transform!(targs..., varlist, cv_gradlist, cc_gradlist)
    end

    # Shrink the equations in new_equations and use the substitution process
    # to simultaneously eliminate auxiliary variables in the subgradient
    # vectors for the final expression
    shrink_grad!(cv_gradlist, cc_gradlist, new_equations, force=force)

    # Return only the subgradients of the final term
    return cv_gradlist[:,end], cc_gradlist[:,end]
end
grad(a::SymbolicUtils.BasicSymbolic; force::Bool=false, expand::Bool=false, constants::Vector{Num}=Num[]) = grad(Num(a), force=force, expand=expand, constants=constants)
grad(a::SymbolicUtils.BasicSymbolic, b::Vector{Num}; force::Bool=false, expand::Bool=false, constants::Vector{Num}=Num[]) = grad(Num(a), b, force=force, expand=expand, constants=constants)

# The subgradients are constructed in sequence, but they are comprised of terms of the McCormick
# tuples of the original variables and new auxiliary variables. Therefore we only need to
# substitute out auxiliary variables in the right-most column of each gradlist, which contains
# the subgradient of the original input `Num` in `grad`.
function shrink_grad!(cv_gradlist::Matrix{Num}, cc_gradlist::Matrix{Num}, eqs::Vector{Equation}; force::Bool=false)
    new_eqs = copy(eqs)
    for _ in 1:length(eqs)-4
        # Perform the same substitution as in shrink_eqs
        lhs = string(new_eqs[1].lhs)
        replace = [false; in.(lhs, [string.(x) for x in pull_vars.(new_eqs[2:end])])]
        replacecount = sum(length.(collect.(eachmatch.(Regex("$(lhs)"), string.(new_eqs[2:end])))))
        # println("Substituting $(length(string(new_eqs[1].rhs))), $(replacecount) times")
        if !force && length(string(new_eqs[1].rhs))*replacecount > 10000000
            @warn """Your expression may be too complicated for SourceCodeMcCormick to handle
            without using substantial CPU memory. Consider breaking your expression
            into smaller components using `all_evaluators` and user-defined code.
            (or use the option `force=true` to force this operation to continue)"""
            return
        end
        new_eqs[replace] = substitute(new_eqs[replace], Dict(new_eqs[1].lhs => new_eqs[1].rhs))

        # But also substitute into the right-most column of the gradlists
        replace = in.(lhs, [string.(x) for x in pull_vars.(cv_gradlist[:,end])])
        replacecount = sum(length.(collect.(eachmatch.(Regex("$(lhs)"), string.(cv_gradlist[:,end])))))
        if !force && length(string(new_eqs[1].rhs))*replacecount > 10000000
            @warn """Your expression may be too complicated for SourceCodeMcCormick to handle
            without using substantial CPU memory. Consider breaking your expression
            into smaller components using `all_evaluators` and user-defined code.
            (or use the option `force=true` to force this operation to continue)"""
            return
        end
        cv_gradlist[replace,end] = substitute(cv_gradlist[replace,end], Dict(new_eqs[1].lhs => new_eqs[1].rhs))

        replace = in.(lhs, [string.(x) for x in pull_vars.(cc_gradlist[:,end])])
        replacecount = sum(length.(collect.(eachmatch.(Regex("$(lhs)"), string.(cc_gradlist[:,end])))))
        if !force && length(string(new_eqs[1].rhs))*replacecount > 10000000
            @warn """Your expression may be too complicated for SourceCodeMcCormick to handle
            without using substantial CPU memory. Consider breaking your expression
            into smaller components using `all_evaluators` and user-defined code.
            (or use the option `force=true` to force this operation to continue)"""
            return
        end
        cc_gradlist[replace,end] = substitute(cc_gradlist[replace,end], Dict(new_eqs[1].lhs => new_eqs[1].rhs))
        new_eqs = new_eqs[2:end]
    end

    # The McCormick object of the final expression may also be needed for gradient calculations, so we might
    # as well try substituting that as well.
    for i in eachindex(new_eqs)
        # Don't shrink new_eqs anymore, only perform substitutions into the gradlists
        lhs = string(new_eqs[i].lhs)
        replace = in.(lhs, [string.(x) for x in pull_vars.(cv_gradlist[:,end])])
        replacecount = sum(length.(collect.(eachmatch.(Regex("$(lhs)"), string.(cv_gradlist[:,end])))))
        if !force && length(string(new_eqs[i].rhs))*replacecount > 10000000
            @warn """Your expression may be too complicated for SourceCodeMcCormick to handle
            without using substantial CPU memory. Consider breaking your expression
            into smaller components using `all_evaluators` and user-defined code.
            (or use the option `force=true` to force this operation to continue)"""
            return
        end
        cv_gradlist[replace,end] = substitute(cv_gradlist[replace,end], Dict(new_eqs[i].lhs => new_eqs[i].rhs))

        replace = in.(lhs, [string.(x) for x in pull_vars.(cc_gradlist[:,end])])
        replacecount = sum(length.(collect.(eachmatch.(Regex("$(lhs)"), string.(cc_gradlist[:,end])))))
        if !force && length(string(new_eqs[i].rhs))*replacecount > 10000000
            @warn """Your expression may be too complicated for SourceCodeMcCormick to handle
            without using substantial CPU memory. Consider breaking your expression
            into smaller components using `all_evaluators` and user-defined code.
            (or use the option `force=true` to force this operation to continue)"""
            return
        end
        cc_gradlist[replace,end] = substitute(cc_gradlist[replace,end], Dict(new_eqs[i].lhs => new_eqs[i].rhs))
    end

    return
end


# Need a function like convex_evaluator, but for subgradients.
convex_subgradient(term::BasicSymbolic; force::Bool=false, constants::Vector{Num}=Num[]) = convex_subgradient(Num(term), force=force, constants=constants)
convex_subgradient(term::Num; force::Bool=false, constants::Vector{Num}=Num[]) = convex_subgradient(term, pull_vars(term), force=force, constants=constants)
function convex_subgradient(term::Num, varlist::Vector{Num}; force::Bool=false, constants::Vector{Num}=Num[])
    if exprtype(term.val) == ADD
        cvgrad = zeros(Num, length(vars))
        for (key,val) in term.val.dict
            new_cvgrad, _ = grad(val*key, varlist, force=force, constants=constants)
            if isnothing(new_cvgrad)
                return
            end
            cvgrad .+= new_cvgrad
        end
        
        ordered_vars = pull_vars(cvgrad)
        func_list = []
        for i = eachindex(cvgrad)
            @eval new_func = $(build_function(cvgrad[i], ordered_vars..., expression=Val{true}))
            push!(func_list, new_func)
        end
    else
        cvgrad, _ = grad(term, varlist, force=force, constants=constants)
        if isnothing(cvgrad)
            return
        end
        ordered_vars = pull_vars(cvgrad)
        func_list = []
        for i = eachindex(cvgrad)
            @eval new_func = $(build_function(cvgrad[i], ordered_vars..., expression=Val{true}))
            push!(func_list, new_func)
        end
    end
    return func_list, ordered_vars
end

all_subgradients(term::BasicSymbolic; force::Bool=false, expand::Bool=false, constants::Vector{Num}=Num[]) = all_subgradients(Num(term), force=force, expand=expand, constants=constants)
all_subgradients(term::Num; force::Bool=false, expand::Bool=false, constants::Vector{Num}=Num[]) = all_subgradients(term, pull_vars(term), force=force, expand=expand, constants=constants)
function all_subgradients(term::Num, varlist::Vector{Num}; force::Bool=false, expand::Bool=false, constants::Vector{Num}=Num[])
    if exprtype(term.val) == ADD
        cvgrad = zeros(Num, length(varlist))
        ccgrad = zeros(Num, length(varlist))
        for (key,val) in term.val.dict
            new_cvgrad, new_ccgrad = grad(val*key, varlist, force=force, expand=expand, constants=constants)
            if isnothing(new_cvgrad)
                return
            end
            cvgrad .+= new_cvgrad
            ccgrad .+= new_ccgrad
        end
        
        ordered_vars = pull_vars(cvgrad + ccgrad)
        cv_func_list = []
        cc_func_list = []
        for i = eachindex(cvgrad)
            @eval new_func = $(build_function(cvgrad[i], ordered_vars..., expression=Val{true}))
            push!(cv_func_list, new_func)
            @eval new_func = $(build_function(ccgrad[i], ordered_vars..., expression=Val{true}))
            push!(cc_func_list, new_func)
        end
    else
        cvgrad, ccgrad = grad(term, varlist, force=force, expand=expand, constants=constants)
        if isnothing(cvgrad)
            return
        end
        ordered_vars = pull_vars(cvgrad + ccgrad)
        cv_func_list = []
        cc_func_list = []
        for i = eachindex(cvgrad)
            @eval new_func = $(build_function(cvgrad[i], ordered_vars..., expression=Val{true}))
            push!(cv_func_list, new_func)
            @eval new_func = $(build_function(ccgrad[i], ordered_vars..., expression=Val{true}))
            push!(cc_func_list, new_func)
        end
    end
    return cv_func_list, cc_func_list, ordered_vars
end

include(joinpath(@__DIR__, "rules.jl"))
