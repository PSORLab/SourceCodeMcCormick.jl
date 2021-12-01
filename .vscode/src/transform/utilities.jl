first(a::Expr) = a.args

arity(a::Expr) = length(a.args) - 1
arity(a::Assignment) = arity(rhs(a))

op(a::Expr) = a.args[1]
op(a::Assignment) = op(rhs(a))

xstr(a::Assignment) = String(rhs(a).args[2])
ystr(a::Assignment) = String(rhs(a).args[3])
zstr(a::Assignment) = String(lhs(a))