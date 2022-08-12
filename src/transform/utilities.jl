
arity(a::Equation) = arity(a.rhs)
arity(a::Term{Real, Base.ImmutableDict{DataType,Any}}) = 1
arity(a::Term{Real, Nothing}) = 1
arity(a::SymbolicUtils.Add) = length(a.dict) + (~iszero(a.coeff))
arity(a::SymbolicUtils.Mul) = length(a.dict) + (~isone(a.coeff))
arity(a::SymbolicUtils.Pow) = 2
arity(a::SymbolicUtils.Div) = 2

op(a::Equation) = op(a.rhs)
op(::SymbolicUtils.Add) = +
op(::SymbolicUtils.Mul) = *
op(::SymbolicUtils.Pow) = ^
op(::SymbolicUtils.Div) = /
op(::Term{Real, Base.ImmutableDict{DataType,Any}}) = nothing
op(a::Term{Real, Nothing}) = a.f

xstr(a::Equation) = sub_1(a.rhs)
ystr(a::Equation) = sub_2(a.rhs)
zstr(a::Equation) = a.lhs

sub_1(a::Term{Real, Base.ImmutableDict{DataType,Any}}) = a
function sub_1(a::SymbolicUtils.Add)
    sorted_dict = sort(collect(a.dict), by=x->string(x[1]))
    return sorted_dict[1].first
end
function sub_2(a::SymbolicUtils.Add)
    ~(iszero(a.coeff)) && return a.coeff
    sorted_dict = sort(collect(a.dict), by=x->string(x[1]))
    return sorted_dict[2].first
end

function sub_1(a::SymbolicUtils.Mul)
    sorted_dict = sort(collect(a.dict), by=x->string(x[1]))
    return sorted_dict[1].first
end
function sub_2(a::SymbolicUtils.Mul)
    ~(isone(a.coeff)) && return a.coeff
    sorted_dict = sort(collect(a.dict), by=x->string(x[1]))
    return sorted_dict[2].first
end
function sub_1(a::SymbolicUtils.Div)
    return a.num
end
function sub_2(a::SymbolicUtils.Div)
    return a.den
end
function sub_1(a::SymbolicUtils.Pow)
    return a.base
end
function sub_2(a::SymbolicUtils.Pow)
    return a.exp
end

function sub_1(a::Term{Real, Nothing})
    if a.f==getindex
        return a
    else
        return a.arguments[1]
    end
end
sub_2(a::Term{Real, Nothing}) = a.arguments[2]


# Uses Symbolics functions to generate a variable as a function of the dependent variables of choice (default: t)
function genvar(a::Symbol)
    @isdefined(t) ? genvar(a, :t) : genparam(a)
end
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
function genparam(a::Symbol)
    params = Symbol[]
    ex = Expr(:block)
    var_name, expr = Symbolics.construct_vars(:parameters, a, Real, nothing, nothing, nothing, ModelingToolkit.toparam, false)
    push!(params, var_name)
    push!(ex.args, expr)
    rhs = Symbolics.build_expr(:vect, params)
    push!(ex.args, rhs)
    eval(ex)[1]
end


function extract_terms(eqs::Vector{Equation})
    allstates = SymbolicUtils.OrderedSet()
    ps = SymbolicUtils.OrderedSet()
    for eq in eqs
        if ~(eq.lhs isa Number)
            iv = ModelingToolkit.iv_from_nested_derivative(eq.lhs)
            break
        end
    end
    iv = ModelingToolkit.value(iv)
    for eq in eqs
        eq.lhs isa Union{SymbolicUtils.Symbolic,Number} || (push!(compressed_eqs, eq); continue)
        ModelingToolkit.collect_vars!(allstates, ps, eq.lhs, iv)
        ModelingToolkit.collect_vars!(allstates, ps, eq.rhs, iv)
    end

    return allstates, ps
end

# Interprets existing start points in an ODESystem, then applies the provided bounds to the
# given term and adds to (or overwrites) existing start points. Returns an updated ODESystem
function set_bounds(sys::ODESystem, term::Num, bounds::Tuple{Float64, Float64})
    base_name = get_name(Symbolics.value(term))
    name_lo = String(base_name)*"_"*"lo"
    name_hi = String(base_name)*"_"*"hi"

    model_terms = Vector{Union{Term,Sym}}()
    for i in sys.states
        push!(model_terms, Symbolics.value(i))
    end
    for i in sys.ps
        push!(model_terms, Symbolics.value(i))
    end
    real_lo = nothing
    real_hi = nothing
    for i in model_terms
        if String(get_name(i))==name_lo
            real_lo = i
        elseif String(get_name(i))==name_hi
            real_hi = i
        end
    end
    if real_lo in keys(sys.defaults)
        delete!(sys.defaults, real_lo)
        sys.defaults[real_lo] = bounds[1]
    else
        sys.defaults[real_lo] = bounds[1]
    end
    if real_hi in keys(sys.defaults)
        delete!(sys.defaults, real_hi)
        sys.defaults[real_hi] = bounds[2]
    else
        sys.defaults[real_hi] = bounds[2]
    end
    return sys
end

function set_bounds(sys::ODESystem, terms::Vector{Num}, bounds::Vector{Tuple{Float64, Float64}})
    for i in 1:length(terms)
        sys = set_bounds(sys, terms[i], bounds[i])
    end
    return sys
end 

function get_cvcc_start_dict(sys::ODESystem, term::Num, start_point::Float64)
    base_name = get_name(Symbolics.value(term))
    name_cv = String(base_name)*"_"*"cv"
    name_cc = String(base_name)*"_"*"cc"

    model_terms = Vector{Union{Term,Sym}}()
    for i in sys.states
        push!(model_terms, Symbolics.value(i))
    end
    for i in sys.ps
        push!(model_terms, Symbolics.value(i))
    end
    real_cv = nothing
    real_cc = nothing
    for i in model_terms
        if String(get_name(i))==name_cv
            real_cv = i
        elseif String(get_name(i))==name_cc
            real_cc = i
        end
    end

    new_dict = copy(sys.defaults)
    if real_cv in keys(new_dict)
        delete!(new_dict, real_cv)
        new_dict[real_cv] = start_point
    else
        new_dict[real_cv] = start_point
    end
    if real_cc in keys(new_dict)
        delete!(new_dict, real_cc)
        new_dict[real_cc] = start_point
    else
        new_dict[real_cc] = start_point
    end
    return new_dict
end


# Side note: this is how you can get a and b to show up NOT as a(t) and b(t)
# t = genparam(:t)
# a = genvar(:a)
# b = genvar(:b)
# st = SymbolicUtils.Code.NameState(Dict(a => :a, b => :b))
# toexpr(a+b, st)


"""
    pull_vars(eqn::Equation)
    pull_vars(eqns::Vector{Equation})

Pull out all variables/symbols from the RHS of an equation or set
of equations and sorts them alphabetically. 

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
function pull_vars(eqn::Equation)
    vars = Num[]
    strings = String[]
    vars, strings = _pull_vars(eqn.rhs, vars, strings)
    vars = vars[sortperm(strings)]
    return vars
end

function pull_vars(eqns::Vector{Equation})
    vars = Num[]
    strings = String[]
    for eqn in eqns
        vars, strings = _pull_vars(eqn.rhs, vars, strings)
    end
    vars = vars[sortperm(strings)]
    return vars
end

function _pull_vars(term::SymbolicUtils.Add, vars::Vector{Num}, strings::Vector{String})
    args = arguments(term)
    for arg in args
        if (typeof(arg) <: Sym{Real, Base.ImmutableDict{DataType, Any}})
            if ~(string(arg) in strings)
                push!(strings, string(arg))
                push!(vars, arg)
            end
        elseif (typeof(arg) <: Int) || (typeof(arg) <: AbstractFloat)
            nothing
        else
            vars, strings = _pull_vars(arg, vars, strings)
        end
    end
    return vars, strings
end

function _pull_vars(term::SymbolicUtils.Mul, vars::Vector{Num}, strings::Vector{String})
    args = arguments(term)
    for arg in args
        if (typeof(arg) <: Sym{Real, Base.ImmutableDict{DataType, Any}})
            if ~(string(arg) in strings)
                push!(strings, string(arg))
                push!(vars, arg)
            end
        elseif (typeof(arg) <: Int) || (typeof(arg) <: AbstractFloat)
            nothing
        else
            vars, strings = _pull_vars(arg, vars, strings)
        end
    end
    return vars, strings
end

function _pull_vars(term::SymbolicUtils.Div, vars::Vector{Num}, strings::Vector{String})
    args = arguments(term)
    for arg in args
        if (typeof(arg) <: Sym{Real, Base.ImmutableDict{DataType, Any}})
            if ~(string(arg) in strings)
                push!(strings, string(arg))
                push!(vars, arg)
            end
        elseif (typeof(arg) <: Int) || (typeof(arg) <: AbstractFloat)
            nothing
        else
            vars, strings = _pull_vars(arg, vars, strings)
        end
    end
    return vars, strings
end

function _pull_vars(term::SymbolicUtils.Pow, vars::Vector{Num}, strings::Vector{String})
    args = arguments(term)
    for arg in args
        if (typeof(arg) <: Sym{Real, Base.ImmutableDict{DataType, Any}})
            if ~(string(arg) in strings)
                push!(strings, string(arg))
                push!(vars, arg)
            end
        elseif (typeof(arg) <: Int) || (typeof(arg) <: AbstractFloat)
            nothing
        else
            vars, strings = _pull_vars(arg, vars, strings)
        end
    end
    return vars, strings
end

function _pull_vars(term::SymbolicUtils.Term{Real, Nothing}, vars::Vector{Num}, strings::Vector{String})
    args = arguments(term)
    for arg in args
        if (typeof(arg) <: Sym{Real, Base.ImmutableDict{DataType, Any}})
            if ~(string(arg) in strings)
                push!(strings, string(arg))
                push!(vars, arg)
            end
        elseif (typeof(arg) <: Int) || (typeof(arg) <: AbstractFloat)
            nothing
        else
            vars, strings = _pull_vars(arg, vars, strings)
        end
    end
    return vars, strings
end

function _pull_vars(term::SymbolicUtils.Term{Bool, Nothing}, vars::Vector{Num}, strings::Vector{String})
    args = arguments(term)
    for arg in args
        if (typeof(arg) <: Sym{Real, Base.ImmutableDict{DataType, Any}})
            if ~(string(arg) in strings)
                push!(strings, string(arg))
                push!(vars, arg)
            end
        elseif (typeof(arg) <: Int) || (typeof(arg) <: AbstractFloat)
            nothing
        else
            vars, strings = _pull_vars(arg, vars, strings)
        end
    end
    return vars, strings
end

function _pull_vars(term::SymbolicUtils.Term{Float64, Nothing}, vars::Vector{Num}, strings::Vector{String})
    return vars, strings
end

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
shrink_eqs(eqs, 1)

1-element Vector{Equation}:
 z ~ (1 + 15x)^2
```
"""
function shrink_eqs(eqs::Vector{Equation}, keep::Int64=4)
    new_eqs = eqs
    for _ in 1:length(eqs)-keep
        new_eqs = substitute(new_eqs, Dict(new_eqs[1].lhs => new_eqs[1].rhs))[2:end]
    end
    return new_eqs
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
x_cv = CUDA.rand(10000)
x_cc = CUDA.rand(10000)
x_hi = CUDA.ones(10000)
x_lo = CUDA.zeros(10000)
y_cv = CUDA.rand(10000)
y_cc = CUDA.rand(10000)
y_hi = CUDA.ones(10000)
y_lo = CUDA.zeros(10000)
out = evaluator.(x_cc, x_cv, x_hi, x_lo, y_cc, y_cv, y_hi, y_lo)
as_array = Array(out)
```
"""
function convex_evaluator(term::Num)
    # First, check to see if the term is "Add". If so, we can get some
    # huge time savings by separating out the expression using the knowledge
    # that the sum of convex relaxations is equal to the convex relaxation
    # of the sum (i.e., a_cv + b_cv = (a+b)_cv, and same for lo/hi/cc)
    if typeof(term.val) <: SymbolicUtils.Add
        # Start with any real-valued operands [if present]
        cv_eqn = term.val.coeff

        # Loop through the dictionary of operands and treat each term like
        # its own equation
        for (key,val) in term.val.dict
            # "key" is the operand, "val" is its coefficient. The LHS of "equation" is irrelevant
            equation = 0 ~ (val*key)

            # Apply the McCormick transform to expand out the equation with auxiliary
            # variables and get expressions for each variable's relaxations
            step_1 = apply_transform(McCormickIntervalTransform(), [equation])

            # Shrink the equations down to 4 total, for "lo", "hi", "cv", and "cc"
            step_2 = shrink_eqs(step_1)

            # For "convex_evaluator" we only care about the convex part, which is #3 of 4.
            # See "all_evaluators" if you need more than just the convex relaxation
            cv_eqn += step_2[3].rhs
        end

        # Scan through the equation and pick out and organize all variables needed as inputs
        ordered_vars = pull_vars(0 ~ cv_eqn)

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
        step_1 = apply_transform(McCormickIntervalTransform(), [equation])
        step_2 = shrink_eqs(step_1)
        ordered_vars = pull_vars(step_2)
        @eval new_func = $(build_function(step_2[3].rhs, ordered_vars..., expression=Val{true}))
    end
    return new_func, ordered_vars
end

function convex_evaluator(equation::Equation)
    # Same as when the input is `Num`, but we have to deal with the input
    # already being an equation (whose LHS is irrelevant)
    if typeof(equation.rhs.val) <: SymbolicUtils.Add
        cv_eqn = equation.rhs.val.coeff
        for (key,val) in equation.rhs.val.dict
            new_equation = 0 ~ (val*key)
            step_1 = apply_transform(McCormickIntervalTransform(), [new_equation])
            step_2 = shrink_eqs(step_1)
            cv_eqn += step_2[3].rhs
        end
        ordered_vars = pull_vars(0~cv_eqn)
        @eval new_func = $(build_function(cv_eqn, ordered_vars..., expression=Val{true}))

    else
        step_1 = apply_transform(McCormickIntervalTransform(), [equation])
        step_2 = shrink_eqs(step_1)
        ordered_vars = pull_vars(step_2)
        @eval new_func = $(build_function(step_2[3].rhs, ordered_vars..., expression=Val{true}))
    end

    return new_func, ordered_vars
end

"""
    all_evaluators(::Num)
    all_evaluators(::Equation)

See `convex_evaluator`. This function performs the same task, but returns
four functions (representing lower bound, upper bound, convex relaxation,
and concave relaxation evaluation functions) and the order vector.
"""
function all_evaluators(term::Num)
    if typeof(term.val) <: SymbolicUtils.Add
        lo_eqn = term.val.coeff
        hi_eqn = term.val.coeff
        cv_eqn = term.val.coeff
        cc_eqn = term.val.coeff
        for (key,val) in term.val.dict
            equation = 0 ~ (val*key)
            step_1 = apply_transform(McCormickIntervalTransform(), [equation])
            step_2 = shrink_eqs(step_1)
            lo_eqn += step_2[1].rhs
            hi_eqn += step_2[2].rhs
            cv_eqn += step_2[3].rhs
            cc_eqn += step_2[3].rhs
        end
        ordered_vars = pull_vars(step_2)
        @eval lo_evaluator = $(build_function(lo_eqn, ordered_vars..., expression=Val{true}))
        @eval hi_evaluator = $(build_function(hi_eqn, ordered_vars..., expression=Val{true}))
        @eval cv_evaluator = $(build_function(cv_eqn, ordered_vars..., expression=Val{true}))
        @eval cc_evaluator = $(build_function(cc_eqn, ordered_vars..., expression=Val{true}))
    else
        equation = 0 ~ term
        step_1 = apply_transform(McCormickIntervalTransform(), [equation])
        step_2 = shrink_eqs(step_1)
        ordered_vars = pull_vars(step_2)
        @eval lo_evaluator = $(build_function(step_2[1].rhs, ordered_vars..., expression=Val{true}))
        @eval hi_evaluator = $(build_function(step_2[2].rhs, ordered_vars..., expression=Val{true}))
        @eval cv_evaluator = $(build_function(step_2[3].rhs, ordered_vars..., expression=Val{true}))
        @eval cc_evaluator = $(build_function(step_2[4].rhs, ordered_vars..., expression=Val{true}))
    end
    return lo_evaluator, hi_evaluator, cv_evaluator, cc_evaluator, ordered_vars
end
function all_evaluators(equation::Equation)
    if typeof(equation.rhs.val) <: SymbolicUtils.Add
        lo_eqn = equation.rhs.val.coeff
        hi_eqn = equation.rhs.val.coeff
        cv_eqn = equation.rhs.val.coeff
        cc_eqn = equation.rhs.val.coeff
        for (key,val) in equation.rhs.val.dict
            new_equation = 0 ~ (val*key)
            step_1 = apply_transform(McCormickIntervalTransform(), [new_equation])
            step_2 = shrink_eqs(step_1)
            lo_eqn += step_2[1].rhs
            hi_eqn += step_2[2].rhs
            cv_eqn += step_2[3].rhs
            cc_eqn += step_2[3].rhs
        end
        ordered_vars = pull_vars(step_2)
        @eval lo_evaluator = $(build_function(lo_eqn, ordered_vars..., expression=Val{true}))
        @eval hi_evaluator = $(build_function(hi_eqn, ordered_vars..., expression=Val{true}))
        @eval cv_evaluator = $(build_function(cv_eqn, ordered_vars..., expression=Val{true}))
        @eval cc_evaluator = $(build_function(cc_eqn, ordered_vars..., expression=Val{true}))
    else
        step_1 = apply_transform(McCormickIntervalTransform(), [equation])
        step_2 = shrink_eqs(step_1)
        ordered_vars = pull_vars(step_2)
        @eval lo_evaluator = $(build_function(step_2[1].rhs, ordered_vars..., expression=Val{true}))
        @eval hi_evaluator = $(build_function(step_2[2].rhs, ordered_vars..., expression=Val{true}))
        @eval cv_evaluator = $(build_function(step_2[3].rhs, ordered_vars..., expression=Val{true}))
        @eval cc_evaluator = $(build_function(step_2[4].rhs, ordered_vars..., expression=Val{true}))
    end
    return lo_evaluator, hi_evaluator, cv_evaluator, cc_evaluator, ordered_vars
end