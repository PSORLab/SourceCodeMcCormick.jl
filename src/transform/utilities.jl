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