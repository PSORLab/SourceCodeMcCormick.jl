
using IfElse

# Transformation rules should have the general form listed below:
# Check IntervalArithmetic for valid interval bound rules (we won't be able 
# to do correctly rounded stuff on GPUs but otherwise the operators should work out)

#=
Unitary Rules
=#
function transform_rule(::IntervalTransform, nothing, yL, yU, xL, xU)
    rl = Equation(yL, xL)
    ru = Equation(yU, xU)
    return rl, ru
end
function transform_rule(::IntervalTransform, ::typeof(exp), yL, yU, xL, xU)
    rl = Equation(yL, exp(xL))
    ru = Equation(yU, exp(xU))
    return rl, ru
end

#=
Binary Rules
=#
function transform_rule(::IntervalTransform, ::typeof(+), zL, zU, xL, xU, yL, yU)
    rl = Equation(zL, xL + yL)
    ru = Equation(zU, xU + yU)
    return rl, ru
end
function transform_rule(::IntervalTransform, ::typeof(-), zL, zU, xL, xU, yL, yU)
    rl = Equation(zL, xL - yU)
    ru = Equation(zU, xU - yL)
    return rl, ru
end
function transform_rule(::IntervalTransform, ::typeof(*), zL, zU, xL, xU, yL, yU)
    rl = Equation(zL, IfElse.ifelse(yL >= 0.0, 
        IfElse.ifelse(xL >= 0.0, xL*yL,
            IfElse.ifelse(xU <= 0.0, xL*yU, xL*yU)),
        IfElse.ifelse(yU <= 0.0,
            IfElse.ifelse(xL >= 0.0, xU*yL,
                IfElse.ifelse(xU <= 0.0, xU*yU, xU*yL)),
            IfElse.ifelse(xL > 0.0, xU*yL,
                IfElse.ifelse(xU < 0.0, xL*yU, min(xU*yL, xU*yU))))))
    ru = Equation(zU, IfElse.ifelse(yL >= 0.0, 
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
    rl = Equation(zL, min(xL, yL))
    ru = Equation(zU, min(xU, yU))
    return rl, ru
end
function transform_rule(::IntervalTransform, ::typeof(max), zL, zU, xL, xU, yL, yU)
    rl = Equation(zL, max(xL, yL))
    ru = Equation(zU, max(xU, yU))
    return rl, ru
end

# TODO: /, ^