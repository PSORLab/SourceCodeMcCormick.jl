
using IfElse

# Transformation rules should have the general form listed below:
# Check IntervalArithmetic for valid interval bound rules (we won't be able 
# to do correctly rounded stuff on GPUs but otherwise the operators should work out)

#=
Unitary rules
=#
function transform_rule(::IntervalTransform, ::Symbol, yL, yU, xL, xU)
    rl = Assignment(yL, :($xL))
    ru = Assignment(yU, :($xU))
    AssignmentPair(rl, ru)
end
function transform_rule(::IntervalTransform, ::typeof(exp), yL, yU, xL, xU)
    rl = Assignment(yL, :(exp($xL)))
    ru = Assignment(yU, :(exp($xU)))
    AssignmentPair(rl, ru)
end

#=
Binary Rules
=#
function transform_rule(::IntervalTransform, ::typeof(+), zL, zU, xL, xU, yL, yU)
    rl = Assignment(zL, :($xL + $yL))
    ru = Assignment(zU, :($xU + $yU))
    AssignmentPair(rl, ru)
end
function transform_rule(::IntervalTransform, ::typeof(-), zL, zU, xL, xU, yL, yU)
    rl = Assignment(zL, :($xL - $yU))
    ru = Assignment(zU, :($xU - $yL))
    AssignmentPair(rl, ru)
end
function transform_rule(::IntervalTransform, ::typeof(*), zL, zU, xL, xU, yL, yU)
    rl = Assignment(zL, :(ifelse($yL >= 0.0, 
        ifelse($xL >= 0.0, $xL*$yL,
            ifelse($xU <= 0.0, $xL*$yU, $xL*$yU)),
        ifelse($yU <= 0.0,
            ifelse($xL >= 0.0, $xU*$yL,
                ifelse($xU <= 0.0, $xU*$yU, $xU*$yL)),
            ifelse($xL > 0.0, $xU*$yL,
                ifelse($xU < 0.0, $xL*$yU, min($xU*$yL, $xU*$yU)))))))
    ru = Assignment(zU, :(ifelse($yL >= 0.0, 
        ifelse($xL >= 0.0, $xU*$yU,
            ifelse($xU <= 0.0, $xU*$yL, $xU*$yU)),
        ifelse($yU <= 0.0,
            ifelse($xL >= 0.0, $xL*$yU,
                ifelse($xU <= 0.0, $xL*$yL, $xL*$yL)),
            ifelse($xL > 0.0, $xU*$yU,
                ifelse($xU < 0.0, $xL*$yL, max($xL*$yL, $xU*$yU)))))))
    AssignmentPair(rl, ru)
end

# TODO: /, ^