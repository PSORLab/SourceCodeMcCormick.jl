
arity(a::Equation) = arity(a.rhs)
arity(a::Term{Real, Base.ImmutableDict{DataType,Any}}) = 1
arity(a::Term{Real, Nothing}) = length(a.arguments)
arity(a::SymbolicUtils.Add) = length(a.dict) + (~iszero(a.coeff))
arity(a::SymbolicUtils.Mul) = length(a.dict) + (~isone(a.coeff))

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

sub_1(a::Term{Real, Nothing}) = a.arguments[1]
sub_2(a::Term{Real, Nothing}) = a.arguments[2]


# Uses Symbolics functions to generate a variable as a function of the dependent variables of choice (default: t)
function genvar(a::Symbol, b=:t)
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
