
# Transformation rules should have the general form listed below:
# Check IntervalArithmetic for valid interval bound rules (we won't be able 
# to do correctly rounded stuff on GPUs but otherwise the operators should work out)

#=
Unitary rules
=#
function transform_rule(::IntervalTransform, ::typeof(exp), yL, yU, xL, xU)
    rl = Assignment(yL, :(exp($xL)))
    ru = Assignment(yU, :(exp($xU)))
    AssignmentPair(rl, ru)
end
# TODO: -, other stuff?

#=
Binary Rules
=#
function transform_rule(::IntervalTransform, ::typeof(+), zL, zU, xL, xU, yL, yU)
    rl = Assignment(zL, :($xL + $yL))
    ru = Assignment(zU, :($xU + $yU))
    AssignmentPair(rl, ru)
end
# TODO: *, /, ^, -