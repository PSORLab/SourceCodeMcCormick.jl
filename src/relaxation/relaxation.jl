struct McCormickTransform <: AbstractTransform end
struct McCormickIntervalTransform <: AbstractTransform end

function var_names(::McCormickTransform, s::String)
    scv = Symbol(s*"_cv")
    scc = Symbol(s*"_cc")
    scv, scc
end
function var_names(::McCormickIntervalTransform, s::String)
    sL, sU = var_names(IntervalTransform(), s)
    scv, scc = var_names(McCormickTransform(), s)
    sL, sU, scv, scc
end

# A symbolic way of evaluating the line segment between (xL, zL) and (xU, zU) at x (returns Expr).
line_expr(x, xL, xU, zL, zU) = :(ifelse($zU > $zL, ($zL*($xU - $x) + $zU*($x - $xL))/($xU - $xL), $zU))

# A symbolic way of computing the mid of three numbers (returns Expr).
mid_expr(x, y, z) = :(ifelse(($x < $y) && ($y < $z), $y, ifelse(($z < $y) && ($y < $x), $y,
                      ifelse(($y < $x) && ($x < $z), $x, ifelse(($z < $x) && ($x < $y), $x, $z)))))

include(joinpath(@__DIR__, "rules.jl"))