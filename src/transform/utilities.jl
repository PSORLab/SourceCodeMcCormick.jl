first(a::Expr) = a.args

arity(a::Expr) = length(a.args) - 1
arity(a::Assignment) = arity(a.rhs)
arity(a::Number) = 1

op(a::Expr) = a.args[1]
op(a::Assignment) = op(a.rhs)
op(a::Number) = "const"
op(a::Symbol) = "const"

xstr(a::Assignment) = string_or_num(a.rhs.args[2])
ystr(a::Assignment) = string_or_num(a.rhs.args[3])
zstr(a::Assignment) = string_or_num(a.lhs)

string_or_num(a::Expr) = string(a)
string_or_num(a::Symbol) = string(a)
string_or_num(a::Number) = a

# Uses Symbolics functions to generate a variable as a function of the dependent variables of choice (default: t)
function genvar(a::Symbol, b=:t)
    vars = Symbol[]
    ex = Expr(:block)
    var_name, expr = Symbolics.construct_vars(:variables, a, Real, [b], nothing, nothing, identity, false)
    push!(vars, var_name)
    push!(ex.args, expr)
    rhs = Symbolics.build_expr(:vect, vars)
    push!(ex.args, rhs)
    eval(ex)
end
function genvar(a::Symbol, b::Vector{Symbol})
    vars = Symbol[]
    ex = Expr(:block)
    var_name, expr = Symbolics.construct_vars(:variables, a, Real, b, nothing, nothing, identity, false)
    push!(vars, var_name)
    push!(ex.args, expr)
    rhs = Symbolics.build_expr(:vect, vars)
    push!(ex.args, rhs)
    eval(ex)
end
function genparam(a::Symbol)
    params = Symbol[]
    ex = Expr(:block)
    var_name, expr = Symbolics.construct_vars(:parameters, a, Real, nothing, nothing, nothing, ModelingToolkit.toparam, false)
    push!(params, var_name)
    push!(ex.args, expr)
    rhs = Symbolics.build_expr(:vect, params)
    push!(ex.args, rhs)
    eval(ex)
end