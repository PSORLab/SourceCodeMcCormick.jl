    
using IfElse

# Note: McCormick transform is only the convex and concave portions of the transformation,
# so that the interval transforms (which are the same as for a regular Interval Transform)
# are not repeated in the code. Whenever a McCormick transform is required, 
# McCormickIntervalTransform should call both IntervalTransform and McCormickTransform
# to get all 4 of the required transformations


function transform_rule(::McCormickIntervalTransform, rule::Any, yL, yU, ycv, ycc, xL, xU, xcv, xcc)
    rL, rU = transform_rule(IntervalTransform(), rule, yL, yU, xL, xU)
    rcv, rcc = transform_rule(McCormickTransform(), rule, ycv, ycc, yL, yU, xcv, xcc, xL, xU)
    return rL, rU, rcv, rcc
end
function transform_rule(::McCormickIntervalTransform, rule::Any, zL, zU, zcv, zcc, xL, xU, xcv, xcc, yL, yU, ycv, ycc)
    rL, rU = transform_rule(IntervalTransform(), rule, zL, zU, xL, xU, yL, yU)
    rcv, rcc = transform_rule(McCormickTransform(), rule, zL, zU, zcv, zcc, xL, xU, xcv, xcc, yL, yU, ycv, ycc)
    return rL, rU, rcv, rcc
end



#=
Unitary Rules
=#
function transform_rule(::McCormickTransform, ::typeof(exp), yL, yU, ycv, ycc, xL, xU, xcv, xcc)
    mcv = mid_expr(xcv, xcc, xL)
    mcc = mid_expr(xcv, xcc, xU)
    rcv = Assignment(ycv, exp(mcv))
    rcc = Assignment(ycc, line_expr(mcc, xL, xU, yL, yU))
    return rcv, rcc
end

#=
Binary Rules
=#
function transform_rule(::McCormickTransform, ::typeof(+), zL, zU, zcv, zcc, xL, xU, xcv, xcc, yL, yU, ycv, ycc)
    rcv = Assignment(zcv, xcv + ycv)
    rcc = Assignment(zcc, xcc + ycc)
    return rcv, rcc
end


# Rules for multiplication adapted from:
# https://github.com/PSORLab/McCormick.jl/blob/master/src/forward_operators/multiplication.jl
function transform_rule(::McCormickTransform, ::typeof(*), zL, zU, zcv, zcc, xL, xU, xcv, xcc, yL, yU, ycv, ycc)
    rcv = Assignment(zcv, IfElse.ifelse(xL >= 0.0,
        IfElse.ifelse(yL >= 0.0, max(yU*xcv + xU*ycv - xU*yU, yL*xcv + xL*ycv - xL*yL),
            IfElse.ifelse(yU <= 0.0, -max((-yU)*xcv + xU*(-ycv) - xU*(-yU), (-yL)*xcv + xL*(-ycv) - xL*(-yL)), 
                max(yU*xcv + xU*ycv - xU*yU, yL*xcc + xL*ycv - xL*yL))),
        IfElse.ifelse(xU <= 0.0,
            IfElse.ifelse(yL >= 0.0, -max(yU*(-xcv) + (-xU)*ycv - (-xU)*yU, yL*(-xcv) + (-xL)*ycv - (-xL)*yL),
                IfElse.ifelse(yU <= 0.0, max(yU*xcv + xU*ycv - xU*yU, yL*xcv + xL*ycv - xL*yL), 
                    -max(yU*(-xcv) + (-xU)*ycv - (-xU)*yU, yL*(-xcc) + (-xL)*ycv - (-xL)*yL))),
            IfElse.ifelse(yL >= 0.0, max(xU*ycv + yU*xcv - yU*xU, xL*ycc + yL*xcv - yL*xL),
                IfElse.ifelse(yU <= 0.0, -max(xU*(-ycv) + (-yU)*xcv - (-yU)*xU, xL*(-ycc) + (-yL)*xcv - (-yL)*xL), 
                    max(yU*xcv + xU*ycv - xU*yU, yL*xcc + xL*ycc - xL*yL))))))
    rcc = Assignment(zcc, IfElse.ifelse(xL >= 0.0,
        IfElse.ifelse(yL >= 0.0, min(yL*xcc + xU*ycc - xU*yL, yU*xcc + xU*ycc - xL*yU),
            IfElse.ifelse(yU <= 0.0, -min((-yL)*xcc + xU*(-ycc) - xU*(-yL), (-yU)*xcc + xU*(-ycc) - xL*(-yU)), 
                min(yL*xcv + xU*ycc - xU*yL, yU*xcc + xL*ycc - xL*yU))),
        IfElse.ifelse(xU <= 0.0,
            IfElse.ifelse(yL >= 0.0, -min(yL*(-xcc) + (-xU)*ycc - (-xU)*yL, yU*(-xcc) + (-xU)*ycc - (-xL)*yU),
                IfElse.ifelse(yU <= 0.0, min(yL*xcc + xU*ycc - xU*yL, yU*xcc + xU*ycc - xL*yU), 
                    -min(yL*(-xcv) + (-xU)*ycc - (-xU)*yL, yU*(-xcc) + (-xL)*ycc - (-xL)*yU))),
            IfElse.ifelse(yL >= 0.0, min(xL*ycv + yU*xcc - yU*xL, xU*ycc + yL*xcc - yL*xU),
                IfElse.ifelse(yU <= 0.0, -min(xL*(-ycv) + (-yU)*xcc - (-yU)*xL, xU*(-ycc) + (-yL)*xcc - (-yL)*xU), 
                    min(yL*xcv + xU*ycc - xU*yL, yU*xcc + xL*ycv - xL*yU))))))
    return rcv, rcc

    
    # # This is the base if-else tree for the 9 multiplication cases. I think for the transform rule,
    # # the way this code needs to work, the entire tree needs to be available to the solver since
    # # the equations must be static. The choice of how to write the equations can't change at the
    # # beginning, they need to include every possibility. So I don't think I can include tricks to
    # # only write 3 cases and switch between them as necessary?                                    
    # IfElse.ifelse(xL >= 0.0,
    #     IfElse.ifelse(yL >= 0.0, [x+,y+],
    #         IfElse.ifelse(yU <= 0.0, [x+,y-], [x+,ym]))
    #     IfElse.ifelse(xU <= 0.0,
    #         IfElse.ifelse(yL >= 0.0, [x-,y+],
    #             IfElse.ifelse(yU <= 0.0, [x-,y-], [x-,ym]))
    #         IfElse.ifelse(yL >= 0.0, [xm,y+],
    #             IfElse.ifelse(yU <= 0.0, [xm,y-], [xm,ym]))))

    # # [x+,y+]
    # # Normal case of multiply_STD_NS
    # cv = max(yU*xcv + xU*ycv - xU*yU, yL*xcv + xL*ycv - xL*yL)
    # cc = min(yL*xcc + xU*ycc - xU*yL, yU*xcc + xU*ycc - xL*yU)

    # # [x+,y-]
    # # This returns -mult_kernel(x, -y), so put a negative in front, and then
    # # replace do [x+,y+] but replace all y's with -y because y's are secretly negative?
    # cv = -max((-yU)*xcv + xU*(-ycv) - xU*(-yU), (-yL)*xcv + xL*(-ycv) - xL*(-yL))
    # cc = -min((-yL)*xcc + xU*(-ycc) - xU*(-yL), (-yU)*xcc + xU*(-ycc) - xL*(-yU))


    # # [x+,ym]
    # # Different from "normal case", note mix of cc's and cv's
    # cv = max(yU*xcv + xU*ycv - xU*yU, yL*xcc + xL*ycv - xL*yL)
    # cc = min(yL*xcv + xU*ycc - xU*yL, yU*xcc + xL*ycc - xL*yU)


    # # [x-,y+]
    # # This returns -mult_kernel(-x, y). Negative in front, replace x's with -x's
    # cv = -max(yU*(-xcv) + (-xU)*ycv - (-xU)*yU, yL*(-xcv) + (-xL)*ycv - (-xL)*yL)
    # cc = -min(yL*(-xcc) + (-xU)*ycc - (-xU)*yL, yU*(-xcc) + (-xU)*ycc - (-xL)*yU)


    # # [x-,y-]
    # # This returns mult_kernel(-x,-y), but since each term is of the form x_*y_, 
    # # (-x)*(-y)=xy, so no change from the "normal case"
    # cv = max(yU*xcv + xU*ycv - xU*yU, yL*xcv + xL*ycv - xL*yL)
    # cc = min(yL*xcc + xU*ycc - xU*yL, yU*xcc + xU*ycc - xL*yU)


    # # [x-,ym]
    # # This returns -mult_kernel(y,-x). Negative in front, then we're doing [xm, y+], 
    # # but swap x's and y's and replace x's with (-x)'s
    # cv = -max(yU*(-xcv) + (-xU)*ycv - (-xU)*yU, yL*(-xcc) + (-xL)*ycv - (-xL)*yL)
    # cc = -min(yL*(-xcv) + (-xU)*ycc - (-xU)*yL, yU*(-xcc) + (-xL)*ycc - (-xL)*yU)


    # # [xm,y+]
    # # This returns mult_kernel(y, x). So it says, no, do [y+,xm] to be similar to above,
    # # but with x's and y's switched.
    # cv = max(xU*ycv + yU*xcv - yU*xU, xL*ycc + yL*xcv - yL*xL)
    # cc = min(xL*ycv + yU*xcc - yU*xL, xU*ycc + yL*xcc - yL*xU)


    # # [xm,y-]
    # # This one returns -mult_kernel(-y, x). So copy [x+,ym], but swap x's
    # # and y's, then replace y with (-y), and negative in front
    # cv = -max(xU*(-ycv) + (-yU)*xcv - (-yU)*xU, xL*(-ycc) + (-yL)*xcv - (-yL)*xL)
    # cc = -min(xL*(-ycv) + (-yU)*xcc - (-yU)*xL, xU*(-ycc) + (-yL)*xcc - (-yL)*xU)


    # # [xm,ym]
    # cv = max(yU*xcv + xU*ycv - xU*yU, yL*xcc + xL*ycc - xL*yL)
    # cc = min(yL*xcv + xU*ycc - xU*yL, yU*xcc + xL*ycv - xL*yU)

end


#=
TODO: Add other operators. It's probably helpful to break the McCormick overload and McCormick + Interval Outputs
into separate transform_rules since the coupling for the ODEs are one directional and potentially useful.
=#
