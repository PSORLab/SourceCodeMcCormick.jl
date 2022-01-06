
using IfElse

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
    rl = ifelse(yL >= 0.0,
            ifelse(xL >= 0.0,
                Assignment(zL, :($xL*$yL)),
                ifelse(xU <= 0.0,
                    Assignment(zL, :($xL*$yU)),
                    Assignment(zL, :($xL*$yU)))),
            ifelse(yU <= 0.0,
                ifelse(xL >= 0.0,
                    Assignment(zL, :($xU*$yL)),
                    ifelse(xU <= 0.0,
                        Assignment(zL, :($xU*$yU)),
                        Assignment(zL, :($xU*$yL)))),
                ifelse(xL >= 0.0,
                        Assignment(zL, :($xU*$yL)),
                        ifelse(xU <= 0.0,
                            Assignment(zL, :($xL*$yU)),
                            ifelse(xL*yU < xU*yL,
                                Assignment(zL, :($xL*$yU)),
                                Assignment(zL, :($xU*$yL)))))))
    ru = ifelse(yL >= 0.0,
        ifelse(xL >= 0.0,
            Assignment(zU, :($xU*$yU)),
            ifelse(xU <= 0.0,
                Assignment(zU, :($xU*$yL)),
                Assignment(zU, :($xU*$yU)))),
        ifelse(yU <= 0.0,
            ifelse(xL >= 0.0,
                Assignment(zU, :($xL*$yU)),
                ifelse(xU <= 0.0,
                    Assignment(zU, :($xL*$yL)),
                    Assignment(zU, :($xL*$yL)))),
            ifelse(xL >= 0.0,
                    Assignment(zU, :($xU*$yU)),
                    ifelse(xU <= 0.0,
                        Assignment(zU, :($xL*$yL)),
                        ifelse(xL*yL > xU*yU,
                            Assignment(zU, :($xL*$yL)),
                            Assignment(zU, :($xU*$yU)))))))
    AssignmentPair(rl, ru)

    # Same as above, but in normal if-elseif-else logic
    # if yL >= 0.0
    #     if xL >= 0.0
    #         rl = Assignment(zL, :($xL*$yL))
    #         ru = Assignment(zU, :($xU*$yU))
    #     elseif xU <= 0.0
    #         rl = Assignment(zL, :($xL*$yU))
    #         ru = Assignment(zU, :($xU*$yL))
    #     else
    #         rl = Assignment(zL, :($xL*$yU))
    #         ru = Assignment(zU, :($xU*$yU))
    #     end
    # elseif yU <= 0.0
    #     if xL >= 0.0
    #         rl = Assignment(zL, :($xU*$yL))
    #         ru = Assignment(zU, :($xL*$yU))
    #     elseif xU <= 0.0
    #         rl = Assignment(zL, :($xU*$yU))
    #         ru = Assignment(zU, :($xL*$yL))
    #     else
    #         rl = Assignment(zL, :($xU*$yL))
    #         ru = Assignment(zU, :($xL*$yL))
    #     end
    # else
    #     if xL > 0.0
    #         rl = Assignment(zL, :($xU*$yL))
    #         ru = Assignment(zU, :($xU*$yU))
    #     elseif xU < 0.0
    #         rl = Assignment(zL, :($xL*$yU))
    #         ru = Assignment(zU, :($xL*$yL))
    #     else
    #         if xL*yU < xU*yL
    #             rl = Assignment(zL, :($xL*$yU))
    #         else
    #             rl = Assignment(zL, :($xU*$yL))
    #         end
    #         if xL*yL > xU*yU
    #             ru = Assignment(zU, :($xL*$yL))
    #         else
    #             ru = Assignment(zU, :($xU*$yU))
    #         end
    #     end
    # end
end




# TODO: /, ^