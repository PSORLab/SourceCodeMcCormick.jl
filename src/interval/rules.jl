
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
function transform_rule(::IntervalTransform, ::typeof(getindex), yL, yU, xL, xU)
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
    rl = Equation(zL, IfElse.ifelse(yL >= 0.0, #x*pos
        IfElse.ifelse(xL >= 0.0, xL*yL, #pos*pos
            IfElse.ifelse(xU <= 0.0, xL*yU, xL*yU)), #neg*pos, mix*pos
        IfElse.ifelse(yU <= 0.0, #x*neg
            IfElse.ifelse(xL >= 0.0, xU*yL, #pos*neg
                IfElse.ifelse(xU <= 0.0, xU*yU, xU*yL)), #neg*neg, mix*neg
            IfElse.ifelse(xL > 0.0, xU*yL, #(pos)*mix
                IfElse.ifelse(xU < 0.0, xL*yU, min(xL*yU, xU*yL)))))) #(neg)*mix, mix*mix
    ru = Equation(zU, IfElse.ifelse(yL >= 0.0,  #x*pos
        IfElse.ifelse(xL >= 0.0, xU*yU, #pos*pos
            IfElse.ifelse(xU <= 0.0, xU*yL, xU*yU)), #neg*pos, mix*pos
        IfElse.ifelse(yU <= 0.0, #x*neg
            IfElse.ifelse(xL >= 0.0, xL*yU, #pos*neg
                IfElse.ifelse(xU <= 0.0, xL*yL, xL*yL)), #neg*neg, mix*neg
            IfElse.ifelse(xL > 0.0, xU*yU, #pos*mix
                IfElse.ifelse(xU < 0.0, xL*yL, max(xL*yL, xU*yU)))))) #neg*mix, mix*mix
    return rl, ru
end

function transform_rule(::IntervalTransform, ::typeof(/), zL, zU, xL, xU, yL, yU)
    rl = Equation(zL, IfElse.ifelse(yL > 0.0,
        IfElse.ifelse(xL >= 0.0, xL/yU, #y strictly positive
            IfElse.ifelse(xU <= 0.0, xL/yL, xL/yL)),
        IfElse.ifelse(yU < 0.0, #y strictly negative
            IfElse.ifelse(xL >= 0.0, xU/yU,
                IfElse.ifelse(xU <= 0.0, xU/yL, xU/yU)),
            IfElse.ifelse(yL == 0.0, # y contains 0 and at least yL is 0
                IfElse.ifelse(xL >= 0.0, IfElse.ifelse(yU == 0.0, NaN, IfElse.ifelse(xU == 0.0, 0.0, xL/yU)),
                    IfElse.ifelse(xU <= 0.0, IfElse.ifelse(yU == 0.0, NaN, IfElse.ifelse(xL == 0.0, 0.0, NaN)),
                        NaN)),
                IfElse.ifelse(yU == 0.0, #y contains 0 and yU is 0 but not yL
                    IfElse.ifelse(xL >= 0.0, IfElse.ifelse(xU == 0.0, 0.0, NaN),
                        IfElse.ifelse(xU <= 0.0, IfElse.ifelse(xL == 0.0, 0.0, xU/yL),
                            NaN)),
                    NaN)))))
    ru = Equation(zU, IfElse.ifelse(yL > 0.0,
        IfElse.ifelse(xL >= 0.0, xU/yL, #y strictly positive
            IfElse.ifelse(xU <= 0.0, xU/yU, xU/yL)),
        IfElse.ifelse(yU < 0.0, #y strictly negative
            IfElse.ifelse(xL >= 0.0, xL/yL,
                IfElse.ifelse(xU <= 0.0, xL/yU, xL/yU)),
            IfElse.ifelse(yL == 0.0, # y contains 0 and at least yL is 0
                IfElse.ifelse(xL >= 0.0, IfElse.ifelse(yU == 0.0, NaN, IfElse.ifelse(xU == 0.0, 0.0, NaN)),
                    IfElse.ifelse(xU <= 0.0, IfElse.ifelse(yU == 0.0, NaN, IfElse.ifelse(xL == 0.0, 0.0, xU/yU)),
                        NaN)),
                IfElse.ifelse(yU == 0.0, #y contains 0 and yU is 0 but not yL
                    IfElse.ifelse(xL >= 0.0, IfElse.ifelse(xU == 0.0, 0.0, xL/yL),
                        IfElse.ifelse(xU <= 0.0, IfElse.ifelse(xL == 0.0, 0.0, NaN),
                            NaN)),
                        NaN)))))
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
function transform_rule(::IntervalTransform, ::typeof(^), zL, zU, xL, xU, yL, yU)
    ~((typeof(yL) <: Int) || (typeof(yL) <: AbstractFloat)) && error("Symbolic exponents not currently supported.")
    ~(yL == 2) && error("Exponents besides 2 not currently supported")
    rl = Equation(zL, max(min(xU, 0.0), xL)^2)
    ru = Equation(zU, IfElse.ifelse(xU < 0.0, xL, IfElse.ifelse(xL > 0, xU, IfElse.ifelse(abs(xL) >= abs(xU), xL, xU)))^2)
    return rl, ru
end

# TODO: /, ^