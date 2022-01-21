first(a::Expr) = a.args

arity(a::Expr) = length(a.args) - 1
arity(a::Assignment) = arity(a.rhs)
arity(a::Number) = 1
arity(a::SymbolicUtils.Add) = length(a.dict) + (~iszero(a.coeff))
arity(a::SymbolicUtils.Mul) = length(a.dict) + (~isone(a.coeff))

op(a::Expr) = a.args[1]
op(a::Assignment) = op(a.rhs)
op(::Number) = "const"
op(::Symbol) = "const"
op(::SymbolicUtils.Add) = +
op(::SymbolicUtils.Mul) = *
op(::SymbolicUtils.Pow) = ^
op(::SymbolicUtils.Div) = /

xstr(a::Assignment) = sub_1(a.rhs)
ystr(a::Assignment) = sub_2(a.rhs)
zstr(a::Assignment) = a.lhs

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

# xstr(a::Assignment) = string_or_num(a.rhs.args[2])
# ystr(a::Assignment) = string_or_num(a.rhs.args[3])
# zstr(a::Assignment) = string_or_num(a.lhs)

# string_or_num(a::Num) = a
# string_or_num(a::Expr) = string(a)
# string_or_num(a::Symbol) = string(a)
# string_or_num(a::Number) = a

# # A function to identify whether a term... is.... variable or parameter, hm
# function identify(x::Num)

# end




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