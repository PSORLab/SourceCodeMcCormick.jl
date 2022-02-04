
using IfElse

# Transformation rules should have the general form listed below:
# Check IntervalArithmetic for valid interval bound rules (we won't be able 
# to do correctly rounded stuff on GPUs but otherwise the operators should work out)

#=
Unitary Rules
=#
function transform_rule(::IntervalTransform, nothing, yL, yU, xL, xU)
    rl = Assignment(yL, xL)
    ru = Assignment(yU, xU)
    AssignmentPair(rl, ru)
end
function transform_rule(::IntervalTransform, ::typeof(exp), yL, yU, xL, xU)
    rl = Assignment(yL, exp(xL))
    ru = Assignment(yU, exp(xU))
    return rl, ru
end

#=
Binary Rules
=#
function transform_rule(::IntervalTransform, ::typeof(+), zL, zU, xL, xU, yL, yU)
    rl = Assignment(zL, xL + yL)
    ru = Assignment(zU, xU + yU)
    return rl, ru
end
function transform_rule(::IntervalTransform, ::typeof(-), zL, zU, xL, xU, yL, yU)
    rl = Assignment(zL, xL - yU)
    ru = Assignment(zU, xU - yL)
    return rl, ru
end
function transform_rule(::IntervalTransform, ::typeof(*), zL, zU, xL, xU, yL, yU)
    rl = Assignment(zL, IfElse.ifelse(yL >= 0.0, 
        IfElse.ifelse(xL >= 0.0, xL*yL,
            IfElse.ifelse(xU <= 0.0, xL*yU, xL*yU)),
        IfElse.ifelse(yU <= 0.0,
            IfElse.ifelse(xL >= 0.0, xU*yL,
                IfElse.ifelse(xU <= 0.0, xU*yU, xU*yL)),
            IfElse.ifelse(xL > 0.0, xU*yL,
                IfElse.ifelse(xU < 0.0, xL*yU, min(xU*yL, xU*yU))))))
    ru = Assignment(zU, IfElse.ifelse(yL >= 0.0, 
        IfElse.ifelse(xL >= 0.0, xU*yU,
            IfElse.ifelse(xU <= 0.0, xU*yL, xU*yU)),
        IfElse.ifelse(yU <= 0.0,
            IfElse.ifelse(xL >= 0.0, xL*yU,
                IfElse.ifelse(xU <= 0.0, xL*yL, xL*yL)),
            IfElse.ifelse(xL > 0.0, xU*yU,
                IfElse.ifelse(xU < 0.0, xL*yL, max(xL*yL, xU*yU))))))
    return rl, ru
end


function transform_rule(::IntervalTransform, ::typeof(min), zL, zU, xL, xU, yL, yU)
    rl = Assignment(zL, min(xL, yL))
    ru = Assignment(zU, min(xU, yU))
    return rl, ru
end
function transform_rule(::IntervalTransform, ::typeof(max), zL, zU, xL, xU, yL, yU)
    rl = Assignment(zL, max(xL, yL))
    ru = Assignment(zU, max(xU, yU))
    return rl, ru
end

# TODO: /, ^