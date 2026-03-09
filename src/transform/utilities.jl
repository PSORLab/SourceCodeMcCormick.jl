
# Initial feed functions
arity(a::Equation) = arity(a.rhs)
arity(a::Num) = arity(a.val)
op(a::Equation) = op(a.rhs)
op(a::Num) = op(a.val)

# Helpful classification checker to differentiate between terms like
# "exp(x)" where x is a variable, and terms like "y(t)" where y and t
# are both variables
varterm(a::BasicSymbolic) = (typeof(a.f)<:BasicSymbolic || a.f==getindex) ? true : false

# Informational functions
function arity(a::BasicSymbolic)
    exprtype(a)==SYM  && return 1
    exprtype(a)==TERM && return 1
    exprtype(a)==ADD  && return length(a.dict) + (~iszero(a.coeff))
    exprtype(a)==MUL  && return length(a.dict) + (~isone(a.coeff))
    exprtype(a)==POW  && return 2
    exprtype(a)==DIV  && return 2
end
function op(a::BasicSymbolic)
    exprtype(a)==SYM  && return nothing
    exprtype(a)==TERM && return varterm(a) ? nothing : a.f
    exprtype(a)==ADD  && return +
    exprtype(a)==MUL  && return *
    exprtype(a)==POW  && return ^
    exprtype(a)==DIV  && return /
end

# Component extraction functions
xstr(a::Equation) = sub_1(a.rhs)
ystr(a::Equation) = sub_2(a.rhs)
zstr(a::Equation) = a.lhs

function sub_1(a::BasicSymbolic)
    if exprtype(a)==SYM
        return a
    elseif exprtype(a)==TERM
        (varterm(a) || a.f==getindex) && return a
        return a.arguments[1]
    elseif exprtype(a)==ADD
        sorted_dict = sort(collect(a.dict), by=x->string(x[1]))
        return sorted_dict[1].first
    elseif exprtype(a)==MUL
        sorted_dict = sort(collect(a.dict), by=x->string(x[1]))
        return sorted_dict[1].first
    elseif exprtype(a)==DIV
        return a.num
    elseif exprtype(a)==POW
        return a.base
    end
end
function sub_2(a::BasicSymbolic)
    if exprtype(a)==SYM
        return nothing
    elseif exprtype(a)==TERM
        (varterm(a) || a.f==getindex) && return nothing
        return a.arguments[2]
    elseif exprtype(a)==ADD
        ~(iszero(a.coeff)) && return a.coeff
        sorted_dict = sort(collect(a.dict), by=x->string(x[1]))
        return sorted_dict[2].first
    elseif exprtype(a)==MUL
        ~(isone(a.coeff)) && return a.coeff
        sorted_dict = sort(collect(a.dict), by=x->string(x[1]))
        return sorted_dict[2].first
    elseif exprtype(a)==DIV
        return a.den
    elseif exprtype(a)==POW
        return a.exp
    end
end


# Uses Symbolics functions to generate a variable as a function of the dependent variables of choice 
function genvar(a::Symbol, b::Symbol)
    vars = Symbol[]
    ex = Expr(:block)
    var_name, expr = Symbolics.construct_vars(:variables, a, Real, [b], nothing, nothing, identity, false)
    push!(vars, var_name)
    push!(ex.args, expr)
    rhs = Symbolics.build_expr(:vect, vars)
    push!(ex.args, rhs)
    eval(ex)[1]
end
function genvar(a::Symbol, b::Vector{Symbol})
    vars = Symbol[]
    ex = Expr(:block)
    var_name, expr = Symbolics.construct_vars(:variables, a, Real, b, nothing, nothing, identity, false)
    push!(vars, var_name)
    push!(ex.args, expr)
    rhs = Symbolics.build_expr(:vect, vars)
    push!(ex.args, rhs)
    eval(ex)[1]
end

# If no variables are given, instead create a parameter
genvar(a::Symbol) = genparam(a)
function genparam(a::Symbol)
    params = Symbol[]
    ex = Expr(:block)
    var_name, expr = Symbolics.construct_vars(:parameters, a, Real, nothing, nothing, nothing, identity, false)
    push!(params, var_name)
    push!(ex.args, expr)
    rhs = Symbolics.build_expr(:vect, params)
    push!(ex.args, rhs)
    eval(ex)[1]
end


# # A function to extract terms from a set of equations, for use in dynamic systems
# function extract_terms(eqs::Vector{Equation})
#     allstates = SymbolicUtils.OrderedSet()
#     ps = SymbolicUtils.OrderedSet()
#     for eq in eqs
#         if ~(eq.lhs isa Number)
#             iv = iv_from_nested_derivative(eq.lhs)
#             break
#         end
#     end
#     iv = Symbolics.value(iv)
#     for eq in eqs
#         eq.lhs isa Union{SymbolicUtils.Symbolic,Number} || (push!(compressed_eqs, eq); continue)
#         collect_vars!(allstates, ps, eq.lhs, iv)
#         collect_vars!(allstates, ps, eq.rhs, iv)
#     end

#     return allstates, ps
# end

# # Interprets existing start points in an ODESystem, then applies the provided bounds to the
# # given term and adds to (or overwrites) existing start points. Returns an updated ODESystem
# function set_bounds(sys::ODESystem, term::Num, bounds::Tuple{Float64, Float64})
#     base_name = get_name(Symbolics.value(term))
#     name_lo = String(base_name)*"_"*"lo"
#     name_hi = String(base_name)*"_"*"hi"

#     model_terms = Vector{Union{Term,Sym}}()
#     for i in sys.states
#         push!(model_terms, Symbolics.value(i))
#     end
#     for i in sys.ps
#         push!(model_terms, Symbolics.value(i))
#     end
#     real_lo = nothing
#     real_hi = nothing
#     for i in model_terms
#         if String(get_name(i))==name_lo
#             real_lo = i
#         elseif String(get_name(i))==name_hi
#             real_hi = i
#         end
#     end
#     if real_lo in keys(sys.defaults)
#         delete!(sys.defaults, real_lo)
#         sys.defaults[real_lo] = bounds[1]
#     else
#         sys.defaults[real_lo] = bounds[1]
#     end
#     if real_hi in keys(sys.defaults)
#         delete!(sys.defaults, real_hi)
#         sys.defaults[real_hi] = bounds[2]
#     else
#         sys.defaults[real_hi] = bounds[2]
#     end
#     return sys
# end

# function set_bounds(sys::ODESystem, terms::Vector{Num}, bounds::Vector{Tuple{Float64, Float64}})
#     for i in 1:length(terms)
#         sys = set_bounds(sys, terms[i], bounds[i])
#     end
#     return sys
# end 

# function get_cvcc_start_dict(sys::ODESystem, term::Num, start_point::Float64)
#     base_name = get_name(Symbolics.value(term))
#     name_cv = String(base_name)*"_"*"cv"
#     name_cc = String(base_name)*"_"*"cc"

#     model_terms = Vector{Union{Term,Sym}}()
#     for i in sys.states
#         push!(model_terms, Symbolics.value(i))
#     end
#     for i in sys.ps
#         push!(model_terms, Symbolics.value(i))
#     end
#     real_cv = nothing
#     real_cc = nothing
#     for i in model_terms
#         if String(get_name(i))==name_cv
#             real_cv = i
#         elseif String(get_name(i))==name_cc
#             real_cc = i
#         end
#     end

#     new_dict = copy(sys.defaults)
#     if real_cv in keys(new_dict)
#         delete!(new_dict, real_cv)
#         new_dict[real_cv] = start_point
#     else
#         new_dict[real_cv] = start_point
#     end
#     if real_cc in keys(new_dict)
#         delete!(new_dict, real_cc)
#         new_dict[real_cc] = start_point
#     else
#         new_dict[real_cc] = start_point
#     end
#     return new_dict
# end


"""
    pull_vars(::Num)
    pull_vars(::Vector{Num})
    pull_vars(::Equation)
    pull_vars(::Vector{Equation})

Pull out all variables/symbols from an expression or the RHS of an
equation (or RHSs of a set of equations), and sort them. Variables
are sorted alphabetically, then in the order [cv, cc, L, U], then
followed by the terms for the subgradient of the convex relaxation
and terms for  the subgradient of the concave relaxation.

# Example

```julia> @variables out, x, y, z
julia> func = out ~ z + 2*y - 3*x*z
julia> pull_vars(func)
3-element Vector{Num}:
 x
 y
 z
```
"""
pull_vars(term::BasicSymbolic) = pull_vars(Num(term))
function pull_vars(term::Num)
    vars = Num[]
    strings = String[]
    if ~(typeof(term.val) <: Real)
        vars, strings = _pull_vars(term.val, vars, strings)
        vars = vars[sort_vars(strings)]
    end
    return vars
end

function pull_vars(terms::Vector{Num})
    vars = Num[]
    strings = String[]
    for term in terms
        if ~(typeof(term.val) <: Real)
            vars, strings = _pull_vars(term.val, vars, strings)
        end
    end
    if ~isempty(vars)
        vars = vars[sort_vars(strings)]
    end
    return vars
end

function pull_vars(eqn::Equation)
    vars = Num[]
    strings = String[]
    if ~(typeof(eqn.rhs) <: Real)
        vars, strings = _pull_vars(eqn.rhs, vars, strings)
        vars = vars[sort_vars(strings)]
    end
    return vars
end

function pull_vars(eqns::Vector{Equation})
    vars = Num[]
    strings = String[]
    for eqn in eqns
        if ~(typeof(eqn.rhs) <: Real)
            vars, strings = _pull_vars(eqn.rhs, vars, strings)
        end
    end
    if ~isempty(vars)
        vars = vars[sort_vars(strings)]
    end
    return vars
end
function pull_vars(eqn::T) where T<:Real
    return Num[]
end

# Sorts variables in a more logical ordering, to be consistent
# with McCormick.jl organization. 
function sort_vars(strings::Vector{String})
    # isempty(strings) && return
    sort_names = fill("", length(strings))

    # Step 1) Check for derivative-type variables
    # @show strings
    # split_strings = string.(hcat(split.(strings, "_")...)[1,:])
    split_strings = first.(split.(strings, "_"))
    if strings == split_strings
        # Simpler case; we can sort more-or-less normally

        # Put constants first, if any exist
        for i in eachindex(split_strings)
            if split_strings[i]=="constant"
                split_strings[i] = "_____constant"
            end
        end

        # Sort split_strings, and if the strings follow the pattern [letters][numbers], 
        # sort by [letters] first and then by [numbers]. Otherwise, treat the string
        # just as a normal string
        return sortperm(split_strings, by = s -> begin
                m = match(r"([a-zA-Z]+)(\d+)", s)
                if m !== nothing
                    prefix = m.captures[1]
                    number = parse(Int, m.captures[2])
                    return (prefix, number)
                else
                    return (s, 0)
                end
            end)
    end
    deriv = fill(false, length(strings))
    # Here's a way to check for derivatives if we need to go back to "d" instead of '∂'
    # for i in eachindex(split_strings) 
    #     for j in eachindex(split_strings)
    #         if length(split_strings[j])==1 || split_strings[j][1] != '∂'
    #             continue
    #         end
    #         if length(split_strings[j]) <= length(split_strings[i])
    #             continue
    #         end
    #         if "∂"*split_strings[i] == split_strings[j][1:length(split_strings[i])+3]
    #             deriv[j] = true
    #         end
    #     end
    # end
    for i in eachindex(split_strings)
        if first(split_strings[i]) == '∂'
            deriv[i] = true
        end
    end

    # Step 2) Determine which main variables are involved
    vars = []
    for i in eachindex(split_strings)
        if deriv[i]==false && ~(split_strings[i] in vars)
            push!(vars, split_strings[i])
        end
    end
    if isempty(vars) # Then all are probably derivatives and we need to read vars another way
        for i in eachindex(split_strings)
            new_vars = split(split_strings[i], "∂")[2:end]
            for var in new_vars
                if ~(var in vars)
                    push!(vars, var)
                end
            end
        end
    end

    # Step 3) Attach simplified variable names to each element, and then
    #         add a number identifier for the pattern: [1,2,3,4] = [cv,cc,lo,hi]
    for i in eachindex(split_strings)
        if ~(deriv[i]) && (split_strings[i] in vars)
            sort_names[i] *= split_strings[i]
            if length(strings[i]) == 1
                sort_names[i]*="_0"
            elseif strings[i][end-1:end] == "cv"
                sort_names[i]*="_1"
            elseif strings[i][end-1:end] == "cc"
                sort_names[i]*="_2"
            elseif strings[i][end-1:end] == "lo"
                sort_names[i]*="_3"
            elseif strings[i][end-1:end] == "hi"
                sort_names[i]*="_4"
            else
                sort_names[i]*="_0"
            end
        elseif deriv[i]
            var1 = ""
            var2 = ""
            for j in vars
                if length(j) < length(split_strings[i])
                    string_indices = collect(eachindex(split_strings[i]))
                    if split_strings[i][string_indices[1:length(j)+1]] == "∂"*j
                        var1 = j
                        var2 = split_strings[i][string_indices[length(j)+2:end]]
                    end
                end
            end
            sort_names[i] *= var1
            if strings[i][end-1:end] == "cv"
                sort_names[i]*="__1"
            elseif strings[i][end-1:end] == "cc"
                sort_names[i]*="__2"
            else
                error("Something happened with the name scheme. Please submit an issue.")
            end
            sort_names[i] *= "_"*var2
        end
    end

    # Step 4) If there is a unique variable "constant", make it appear first
    for i in eachindex(sort_names)
        if sort_names[i]=="constant_0"
            sort_names[i] = "_____constant"
        end
    end

    # Step 5) Perform the sort and return the correct ordering
    order = sortperm(sort_names)

    return order
end

function _pull_vars(term::BasicSymbolic, vars::Vector{Num}, strings::Vector{String})
    if exprtype(term)==SYM
        if ~(string(get_name(term)) in strings)
            push!(strings, string(get_name(term)))
            push!(vars, term)
            return vars, strings
        end
        return vars, strings
    end
    if exprtype(term)==TERM && varterm(term)
        if ~(string(term.f) in strings) && (term.f==getindex && ~(string(term) in string.(vars)))
            push!(strings, string(get_name(term)))
            push!(vars, term)
            return vars, strings
        end
        return vars, strings
    end
    args = arguments(term)
    for arg in args
        if typeof(arg)<:Num
            arg = arg.val
        end
        ~(typeof(arg)<:BasicSymbolic) ? continue : nothing
        if exprtype(arg)==SYM
            if ~(string(get_name(arg)) in strings)
                push!(strings, string(get_name(arg)))
                push!(vars, arg)
            end
        elseif typeof(arg) <: Real
            nothing
        else
            vars, strings = _pull_vars(arg, vars, strings)
        end
    end
    return vars, strings
end
# _pull_vars(term, vars::Vector{Num}, strings::Vector{String}) = vars, strings



"""
    shrink_eqs(::Vector{Equation})
    shrink_eqs(::Vector{Equation}, ::Int)

Given a set of symbolic equations, progressively substitute
the RHS definitions of LHS terms until there are only a set
number of equations remaining (default = 4).

# Example

```
eqs = [y ~ 15*x,
       z ~ (1+y)^2]

julia> shrink_eqs(eqs, 1)
1-element Vector{Equation}:
 z ~ (1 + 15x)^2
```
"""
function shrink_eqs(eqs::Vector{Equation}, keep::Int64=4; force::Bool=false)
    new_eqs = copy(eqs)
    for _ in 1:length(eqs)-keep
        lhs = string(new_eqs[1].lhs)
        replace = [false; in.(lhs, [string.(x) for x in pull_vars.(new_eqs[2:end])])]
        replacecount = sum(length.(collect.(eachmatch.(Regex("$(lhs)"), string.(new_eqs[2:end])))))
        if !force && length(string(new_eqs[1].rhs))*replacecount > 10000000
            @warn """Your expression may be too complicated for SourceCodeMcCormick to handle
            without using substantial CPU memory. Consider breaking your expression
            into smaller components using `all_evaluators` and user-defined code.
            (or use the option `force=true` to force this operation to continue)"""
            return
        end
        new_eqs[replace] = substitute(new_eqs[replace], Dict(new_eqs[1].lhs => new_eqs[1].rhs))
        new_eqs = new_eqs[2:end]
    end
    # Need to add in the final shrinking for the cut.
    return new_eqs
end

"""
    extract(::Vector{Equation})
    extract(::Vector{Equation}, ::Int)

Given a set of symbolic equations, and optinally a specific
element that you would like expanded into a full expression,
progressively substitute the RHS definitions of LHS terms 
until there is only that equation remaining (default = end).
Returns the RHS of that expression as an object of type
SymbolicUtils.BasicSymbolic{Real}.

# Example

```
eqs = [y ~ 15*x,
   z ~ (1+y)^2]

julia> extract(eqs)
(1 + 15x)^2
```
"""
function extract(eqs::Vector{Equation}, ID::Int=length(eqs))
    final_expr = eqs[ID].rhs
    progress = true
    while progress
        progress = false
        for var in pull_vars(final_expr)
            eq_ID = findfirst(x -> isequal(x.lhs, var), eqs)
            if !isnothing(eq_ID)
                if isequal(eqs[eq_ID].lhs, eqs[eq_ID].rhs)
                    nothing
                else
                    final_expr = substitute(final_expr, Dict(eqs[eq_ID].lhs => eqs[eq_ID].rhs))
                    progress = true
                end
            end
        end
    end
    return final_expr
end

"""
    convex_evaluator(::Num)
    convex_evaluator(::Equation)

Given a symbolic expression or equation, return a function that evaluates
the convex relaxation of the expression or the equation's right-hand side
and a list of correctly ordered arguments to this new function. To get
evaluator functions for {lower bound, upper bound, convex relaxation,
concave relaxation}, use `all_evaluators`.

# Example

A representative use case is as follows:
```
@variables x, y
to_evaluate = 1 + x*y
evaluator, ordering = convex_evaluator(to_evaluate)
```

The same could be accomplished if `to_evaluate` were an Equation such as
`0 ~ 1 + x*y`, although it is important to note that the left-hand side
of such an equation is irrelevant for this function. In this example,
the "ordering" object is now the vector `Num[xcc, xcv, xhi, xlo, ycc, ycv, yhi, ylo]`,
indicating the correct arguments and argument order to give to `evaluator`.
The names of these arguments are dependent on the variables used in the 
`to_evaluate` expression and are, in general, the variable name(s) with an 
appended `cv` and `cc` for the convex/concave values of the variables 
(i.e., the point you wish to evaluate), `hi` for the variable's upper 
bound, and `lo` for the variable's lower bound. Variables such as `x[1]` 
will instead be represented as `x_1_cv` or equivalent. 

The expression's convex relaxation can now be evaluated at a specific point 
`(x,y)` by setting `x = xcv = xcc` and `y = ycv = ycc`, and `x` and `y`'s upper
and lower bounds to `xhi`, `yhi`, and `xlo`, `ylo`, respectively. E.g., 
to evaluate `to_evaluate`'s convex relaxation on the box `[0, 5]*[1, 3]`
at the point `(2.5, 2)`, you can type:
`evaluator(2.5, 2.5, 5.0, 0.0, 2.0, 2.0, 3.0, 1.0)`.

The `evaluator` function can also be broadcast, including over CuArrays.
E.g., to evaluate the convex relaxation of the `to_evaluate` expression
at 10,000 random points on the box `[0, 1]*[0, 1]` using a GPU, you could type:
```
x_cv = CUDA.rand(Float64, 10000)
x_cc = CUDA.rand(Float64, 10000)
x_hi = CUDA.ones(Float64, 10000)
x_lo = CUDA.zeros(Float64, 10000)
y_cv = CUDA.rand(Float64, 10000)
y_cc = CUDA.rand(Float64, 10000)
y_hi = CUDA.ones(Float64, 10000)
y_lo = CUDA.zeros(Float64, 10000)
out = evaluator.(x_cc, x_cv, x_hi, x_lo, y_cc, y_cv, y_hi, y_lo)
as_array = Array(out)
```
"""
function convex_evaluator(term::Num; force::Bool=false, constants::Vector{Num}=Num[])
    # First, check to see if the term is "Add". If so, we can get some
    # huge time savings by separating out the expression using the knowledge
    # that the sum of convex relaxations is equal to the convex relaxation
    # of the sum (i.e., a_cv + b_cv = (a+b)_cv, and same for lo/hi/cc)
    if exprtype(term.val) == ADD
        # Start with any real-valued operands [if present]
        cv_eqn = term.val.coeff

        # Loop through the dictionary of operands and treat each term like
        # its own equation
        for (key,val) in term.val.dict
            # "key" is the operand, "val" is its coefficient. The LHS of "equation" is irrelevant
            equation = 0 ~ (val*key)

            # Apply the McCormick transform to expand out the equation with auxiliary
            # variables and get expressions for each variable's relaxations
            step_1 = apply_transform(McCormickIntervalTransform(), [equation], constants=constants)

            # Shrink the equations down to 4 total, for "lo", "hi", "cv", and "cc"
            step_2 = shrink_eqs(step_1, force=force)

            # If shrink_eqs fails, back out of the function
            if isnothing(step_2)
                return
            end

            # For "convex_evaluator" we only care about the convex part, which is #3 of 4.
            # See "all_evaluators" if you need more than just the convex relaxation
            cv_eqn += step_2[3].rhs
        end

        # Scan through the equation and pick out and organize all variables needed as inputs
        ordered_vars = pull_vars(Num(cv_eqn))

        # Create the evaluation function. This works by calling Symbolics.build_function,
        # which creates a function as an Expr that evaluates build_function's first
        # argument, with the next argument(s) as the function's input(s). If we
        # set expression=Val{false}, build_function will return a compiled function
        # as a RuntimeGeneratedFunction, which we do NOT want as this is not 
        # GPU-compatible. Instead, we keep expression=Val{true} (technically this is
        # the default) and we set new_func to be the evaluation of the returned Expr,
        # which is now a callable function. This line is delicate--don't change unless
        # you know what you're doing!
        @eval new_func = $(build_function(cv_eqn, ordered_vars..., expression=Val{true}))
    else
        # Same as previous block, but without the speedup from a_cv + b_cv = (a+b)_cv
        equation = 0 ~ term
        step_1 = apply_transform(McCormickIntervalTransform(), [equation], constants=constants)
        step_2 = shrink_eqs(step_1, force=force)
        if isnothing(step_2)
            return
        end
        ordered_vars = pull_vars(step_2)
        @eval new_func = $(build_function(step_2[3].rhs, ordered_vars..., expression=Val{true}))
    end
    return new_func, ordered_vars
end

function convex_evaluator(equation::Equation; force::Bool=false, constants::Vector{Num}=Num[])
    # Same as when the input is `Num`, but we have to deal with the input
    # already being an equation (whose LHS is irrelevant)
    if exprtype(equation.rhs.val) == ADD
        cv_eqn = equation.rhs.val.coeff
        for (key,val) in equation.rhs.val.dict
            new_equation = 0 ~ (val*key)
            step_1 = apply_transform(McCormickIntervalTransform(), [new_equation], constants=constants)
            step_2 = shrink_eqs(step_1, force=force)
            if isnothing(step_2)
                return
            end
            cv_eqn += step_2[3].rhs
        end
        ordered_vars = pull_vars(Num(cv_eqn))
        @eval new_func = $(build_function(cv_eqn, ordered_vars..., expression=Val{true}))

    else
        step_1 = apply_transform(McCormickIntervalTransform(), [equation], constants=constants)
        step_2 = shrink_eqs(step_1, force=force)
        if isnothing(step_2)
            return
        end
        ordered_vars = pull_vars(step_2)
        @eval new_func = $(build_function(step_2[3].rhs, ordered_vars..., expression=Val{true}))
    end

    return new_func, ordered_vars
end

"""
    all_evaluators(::Num)
    all_evaluators(::Equation)

See `convex_evaluator`. This function performs the same task, but returns
four functions (representing functions for the convex relaxation, concave relaxation,
lower bound of the interval extension, and upper bound of the interval extension)
[cv, cc, lo, hi] and the order vector.
"""
function all_evaluators(term::Num; force::Bool=false, constants::Vector{Num}=Num[])
    if exprtype(term.val) == ADD
        lo_eqn = term.val.coeff
        hi_eqn = term.val.coeff
        cv_eqn = term.val.coeff
        cc_eqn = term.val.coeff
        for (key,val) in term.val.dict
            equation = 0 ~ (val*key)
            step_1 = apply_transform(McCormickIntervalTransform(), [equation], constants=constants)
            step_2 = shrink_eqs(step_1, force=force)
            if isnothing(step_2)
                return
            end
            lo_eqn += step_2[1].rhs
            hi_eqn += step_2[2].rhs
            cv_eqn += step_2[3].rhs
            cc_eqn += step_2[4].rhs
        end
        ordered_vars = pull_vars([0~lo_eqn, 0~hi_eqn, 0~cv_eqn, 0~cc_eqn])
        @eval lo_evaluator = $(build_function(lo_eqn, ordered_vars..., expression=Val{true}))
        @eval hi_evaluator = $(build_function(hi_eqn, ordered_vars..., expression=Val{true}))
        @eval cv_evaluator = $(build_function(cv_eqn, ordered_vars..., expression=Val{true}))
        @eval cc_evaluator = $(build_function(cc_eqn, ordered_vars..., expression=Val{true}))
    else
        equation = 0 ~ term
        step_1 = apply_transform(McCormickIntervalTransform(), [equation], constants=constants)
        step_2 = shrink_eqs(step_1, force=force)
        if isnothing(step_2)
            return
        end
        ordered_vars = pull_vars(step_2)
        @eval lo_evaluator = $(build_function(step_2[1].rhs, ordered_vars..., expression=Val{true}))
        @eval hi_evaluator = $(build_function(step_2[2].rhs, ordered_vars..., expression=Val{true}))
        @eval cv_evaluator = $(build_function(step_2[3].rhs, ordered_vars..., expression=Val{true}))
        @eval cc_evaluator = $(build_function(step_2[4].rhs, ordered_vars..., expression=Val{true}))
    end
    return cv_evaluator, cc_evaluator, lo_evaluator, hi_evaluator, ordered_vars
end
function all_evaluators(equation::Equation; force::Bool=false, constants::Vector{Num}=Num[])
    if exprtype(equation.rhs) == ADD
        lo_eqn = equation.rhs.coeff
        hi_eqn = equation.rhs.coeff
        cv_eqn = equation.rhs.coeff
        cc_eqn = equation.rhs.coeff
        for (key,val) in equation.rhs.dict
            new_equation = 0 ~ (val*key)
            step_1 = apply_transform(McCormickIntervalTransform(), [new_equation], constants=constants)
            step_2 = shrink_eqs(step_1, force=force)
            if isnothing(step_2)
                return
            end
            lo_eqn += step_2[1].rhs
            hi_eqn += step_2[2].rhs
            cv_eqn += step_2[3].rhs
            cc_eqn += step_2[4].rhs
        end
        ordered_vars = pull_vars([0~lo_eqn, 0~hi_eqn, 0~cv_eqn, 0~cc_eqn])
        @eval lo_evaluator = $(build_function(lo_eqn, ordered_vars..., expression=Val{true}))
        @eval hi_evaluator = $(build_function(hi_eqn, ordered_vars..., expression=Val{true}))
        @eval cv_evaluator = $(build_function(cv_eqn, ordered_vars..., expression=Val{true}))
        @eval cc_evaluator = $(build_function(cc_eqn, ordered_vars..., expression=Val{true}))
    else
        step_1 = apply_transform(McCormickIntervalTransform(), [equation], constants=constants)
        step_2 = shrink_eqs(step_1, force=force)
        if isnothing(step_2)
            return
        end
        ordered_vars = pull_vars(step_2)
        @eval lo_evaluator = $(build_function(step_2[1].rhs, ordered_vars..., expression=Val{true}))
        @eval hi_evaluator = $(build_function(step_2[2].rhs, ordered_vars..., expression=Val{true}))
        @eval cv_evaluator = $(build_function(step_2[3].rhs, ordered_vars..., expression=Val{true}))
        @eval cc_evaluator = $(build_function(step_2[4].rhs, ordered_vars..., expression=Val{true}))
    end
    return cv_evaluator, cc_evaluator, lo_evaluator, hi_evaluator, ordered_vars
end
