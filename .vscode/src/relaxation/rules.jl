                      
function transform_rule(::McCormickOverload, ::typeof(exp), ycv, ycc, yL, yU, xcv, xcc, xL, xU)
    mcv = mid_expr(xcv, xcc, xL)
    mcc = mid_expr(ycv, ycc, yU)
    rcv = Assignment(ycv, :(exp($mcv)))
    rcc = Assignment(ycv, line_expr(mcc, xL, xU, yL, yU))
    return AssignmentPair(rcv, rcc)
end
function transform_rule(::McCormickIntervalOverload, ::typeof(exp), ycv, ycc, yL, yU, xcv, xcc, xL, xU)
    rL, rU = transform_rule(IntervalOverload(), exp, yL, yU, xL, xU)
    rcv, rcc = transform_rule(McCormickOverload(), exp, ycv, ycc, yL, yU, xcv, xcc, xL, xU)
    return AssignmentQuad(rL, rU, rcv, rcc)
end

#=
TODO: Add other operators. It's probably helpful to break the McCormick overload and McCormick + Interval Outputs
into separate transform_rules since the coupling for the ODEs are one directional and potentially useful.
=#