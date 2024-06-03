
# Note: McCormick transform is only the convex and concave portions of the transformation,
# so that the interval transforms (which are the same as for a regular Interval Transform)
# are not repeated in the code. Whenever a McCormick transform is required, 
# McCormickIntervalTransform should call both IntervalTransform and McCormickTransform
# to get all 4 of the required transformations


function transform_rule(::McCormickIntervalTransform, rule::Any, yL, yU, ycv, ycc, xL, xU, xcv, xcc)
    rL, rU = transform_rule(IntervalTransform(), rule, yL, yU, xL, xU)
    rcv, rcc = transform_rule(McCormickTransform(), rule, yL, yU, ycv, ycc, xL, xU, xcv, xcc)
    return rL, rU, rcv, rcc
end
function transform_rule(::McCormickIntervalTransform, rule::Any, zL, zU, zcv, zcc, xL, xU, xcv, xcc, yL, yU, ycv, ycc)
    rL, rU = transform_rule(IntervalTransform(), rule, zL, zU, xL, xU, yL, yU)
    rcv, rcc = transform_rule(McCormickTransform(), rule, rL.rhs, rU.rhs, zcv, zcc, xL, xU, xcv, xcc, yL, yU, ycv, ycc)
    return rL, rU, rcv, rcc
end



#=
Unitary Rules
=#
function transform_rule(::McCormickTransform, ::Nothing, yL, yU, ycv, ycc, xL, xU, xcv, xcc)
    rcv = Equation(ycv, xcv)
    rcc = Equation(ycc, xcc)
    return rcv, rcc
end
function transform_rule(::McCormickTransform, ::typeof(getindex), yL, yU, ycv, ycc, xL, xU, xcv, xcc)
    rcv = Equation(ycv, xcv)
    rcc = Equation(ycc, xcc)
    return rcv, rcc
end
function transform_rule(::McCormickTransform, ::typeof(exp), yL, yU, ycv, ycc, xL, xU, xcv, xcc)
    mcv = mid_expr(xcv, xcc, xL)
    mcc = mid_expr(xcv, xcc, xU)
    rcv = Equation(ycv, exp(mcv))
    rcc = Equation(ycc, line_expr(mcc, xL, xU, exp(xL), exp(xU)))
    return rcv, rcc
end


#=
Binary Rules
=#
function transform_rule(::McCormickTransform, ::typeof(+), zL, zU, zcv, zcc, xL, xU, xcv, xcc, yL, yU, ycv, ycc)
    rcv = Equation(zcv, xcv + ycv)
    rcc = Equation(zcc, xcc + ycc)
    return rcv, rcc
end


# Rules for multiplication adapted from:
# https://github.com/PSORLab/McCormick.jl/blob/master/src/forward_operators/multiplication.jl
function transform_rule(::McCormickTransform, ::typeof(*), zL, zU, zcv, zcc, xL, xU, xcv, xcc, yL::Real, yU::Real, ycv::Real, ycc::Real)
    rcv = Equation(zcv, IfElse.ifelse(yL >= 0.0, ycv*xcv, ycc*xcc))
    rcc = Equation(zcc, IfElse.ifelse(yL >= 0.0, ycc*xcc, ycv*xcv))
    return rcv, rcc
end
function transform_rule(::McCormickTransform, ::typeof(*), zL, zU, zcv, zcc, xL, xU, xcv, xcc, yL, yU, ycv, ycc)
    rcv = Equation(zcv, max(zL, IfElse.ifelse(xL >= 0.0,
        IfElse.ifelse(yL >= 0.0, max(yU*xcv + xU*ycv - xU*yU, yL*xcv + xL*ycv - xL*yL),
            IfElse.ifelse(yU <= 0.0, -min((-yU)*xcc + xU*(-ycv) - xU*(-yU), (-yL)*xcc + xL*(-ycv) - xL*(-yL)),
                max(yU*xcv + xU*ycv - xU*yU, yL*xcc + xL*ycv - xL*yL))),
        IfElse.ifelse(xU <= 0.0,
            IfElse.ifelse(yL >= 0.0, -min(yL*(-xcv) + (-xL)*ycc - (-xL)*yL, yU*(-xcv) + (-xU)*ycc - (-xU)*yU),
                IfElse.ifelse(yU <= 0.0, max(yL*xcc + xL*ycc - xL*yL, yU*xcc + xU*ycc - xU*yU),
                    -min(yL*(-xcc) + (-xL)*ycc - (-xL)*yL, yU*(-xcv) + (-xU)*ycc - (-xU)*yU))),
            IfElse.ifelse(yL >= 0.0, max(xU*ycv + yU*xcv - yU*xU, xL*ycc + yL*xcv - yL*xL),
                IfElse.ifelse(yU <= 0.0, -min(xL*(-ycc) + (-yL)*xcc - (-yL)*xL, xU*(-ycv) + (-yU)*xcc - (-yU)*xU), 
                max(yU*xcv + xU*ycv - xU*yU, yL*xcc + xL*ycc - xL*yL)))))))
    rcc = Equation(zcc, min(zU, IfElse.ifelse(xL >= 0.0,
        IfElse.ifelse(yL >= 0.0, min(yL*xcc + xU*ycc - xU*yL, yU*xcc + xL*ycc - xL*yU),
            IfElse.ifelse(yU <= 0.0, -max((-yL)*xcv + xU*(-ycc) - xU*(-yL), (-yU)*xcv + xL*(-ycc) - xL*(-yU)),
                min(yL*xcv + xU*ycc - xU*yL, yU*xcc + xL*ycc - xL*yU))),
        IfElse.ifelse(xU <= 0.0,
            IfElse.ifelse(yL >= 0.0, -max(yU*(-xcc) + (-xL)*ycv - (-xL)*yU, yL*(-xcc) + (-xU)*ycv - (-xU)*yL),
                 IfElse.ifelse(yU <= 0.0, min(yU*xcv + xL*ycv - xL*yU, yL*xcv + xU*ycv - xU*yL),
                 -max(yU*(-xcc) + (-xL)*ycv - (-xL)*yU, yL*(-xcv) + (-xU)*ycv - (-xU)*yL))),
            IfElse.ifelse(yL >= 0.0, min(xL*ycv + yU*xcc - yU*xL, xU*ycc + yL*xcc - yL*xU),
                IfElse.ifelse(yU <= 0.0, -max(xU*(-ycc) + (-yL)*xcv - (-yL)*xU, xL*(-ycv) + (-yU)*xcv - (-yU)*xL), 
                    min(yL*xcv + xU*ycc - xU*yL, yU*xcc + xL*ycv - xL*yU)))))))
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
    # cc = min(yL*xcc + xU*ycc - xU*yL, yU*xcc + xL*ycc - xL*yU) #typo fixed

    # # [x+,y-]
    # # This returns -mult_kernel(x, -y), so it's [x+, y+] but with y's all flipped,
    # # and then swap and negate cv/cc
    # cv = -min((-yU)*xcc + xU*(-ycv) - xU*(-yU), (-yL)*xcc + xU*(-ycv) - xL*(-yL))
    # cc = -max((-yL)*xcv + xU*(-ycc) - xU*(-yL), (-yU)*xcv + xL*(-ycc) - xL*(-yU))


    # # [x+,ym]
    # # Different from "normal case", note mix of cc's and cv's
    # cv = max(yU*xcv + xU*ycv - xU*yU, yL*xcc + xL*ycv - xL*yL)
    # cc = min(yL*xcv + xU*ycc - xU*yL, yU*xcc + xL*ycc - xL*yU)


    # # [x-,y+]
    # # This returns -mult_kernel(-x, y). Replace x with flipped x, then swap/negate cv/cc
    # [x+, y+]
    # cv = max(yU*xcv + xU*ycv - xU*yU, yL*xcv + xL*ycv - xL*yL)
    # cc = min(yL*xcc + xU*ycc - xU*yL, yU*xcc + xL*ycc - xL*yU)
    # Then flip x's
    # cv = max(yU*(-xcc) + (-xL)*ycv - (-xL)*yU, yL*(-xcc) + (-xU)*ycv - (-xU)*yL)
    # cc = min(yL*(-xcv) + (-xL)*ycc - (-xL)*yL, yU*(-xcv) + (-xU)*ycc - (-xU)*yU)
    # Then swap/negate cv/cc
    # cv = -min(yL*(-xcv) + (-xL)*ycc - (-xL)*yL, yU*(-xcv) + (-xU)*ycc - (-xU)*yU)
    # cc = -max(yU*(-xcc) + (-xL)*ycv - (-xL)*yU, yL*(-xcc) + (-xU)*ycv - (-xU)*yL)


    # # [x-,y-]
    # # This returns mult_kernel(-x,-y). Swap x with flipped x and y with flipped
    # # y, and since both are flipped, no need to worry about negative signs
    # cv = max(yL*xcc + xL*ycc - xL*yL, yU*xcc + xU*ycc - xU*yU)
    # cc = min(yU*xcv + xL*ycv - xL*yU, yL*xcv + xU*ycv - xU*yL)


    # # [x-,ym]
    # # This returns -mult_kernel(y,-x). So we do [xm, y+] but replace x's with y's,
    # # and y's with flipped x's. Then swap and negate cv/cc
    # This is [xm, y+]
    # cv = max(xU*ycv + yU*xcv - yU*xU, xL*ycc + yL*xcv - yL*xL)
    # cc = min(xL*ycv + yU*xcc - yU*xL, xU*ycc + yL*xcc - yL*xU)
    # Do the swaps:
    # cv = max(yU*(-xcc) + (-xL)*ycv - (-xL)*yU, yL*(-xcv) + (-xU)*ycv - (-xU)*yL)
    # cc = min(yL*(-xcc) + (-xL)*ycc - (-xL)*yL, yU*(-xcv) + (-xU)*ycc - (-xU)*yU)
    # Now swap definitions
    # cv = -min(yL*(-xcc) + (-xL)*ycc - (-xL)*yL, yU*(-xcv) + (-xU)*ycc - (-xU)*yU)
    # cc = -max(yU*(-xcc) + (-xL)*ycv - (-xL)*yU, yL*(-xcv) + (-xU)*ycv - (-xU)*yL)


    # # [xm,y+]
    # # This returns mult_kernel(y, x). So it says, no, do [y+,xm] to be similar to above,
    # # but with x's and y's switched.
    # cv = max(xU*ycv + yU*xcv - yU*xU, xL*ycc + yL*xcv - yL*xL)
    # cc = min(xL*ycv + yU*xcc - yU*xL, xU*ycc + yL*xcc - yL*xU)


    # # [xm,y-]
    # # This one returns -mult_kernel(-y, x). So copy [x+,ym], but replace y with x,
    # and x with (-y), and then swap/negate cv/cc
    # [x+, ym]
    # cv = max(yU*xcv + xU*ycv - xU*yU, yL*xcc + xL*ycv - xL*yL)
    # cc = min(yL*xcv + xU*ycc - xU*yL, yU*xcc + xL*ycc - xL*yU)
    # swap y with x, x with (-y)
    # cv = max(xU*(-ycc) + (-yL)*xcv - (-yL)*xU, xL*(-ycv) + (-yU)*xcv - (-yU)*xL)
    # cc = min(xL*(-ycc) + (-yL)*xcc - (-yL)*xL, xU*(-ycv) + (-yU)*xcc - (-yU)*xU)
    # And now swap/negate
    # cv = -min(xL*(-ycc) + (-yL)*xcc - (-yL)*xL, xU*(-ycv) + (-yU)*xcc - (-yU)*xU)
    # cc = -max(xU*(-ycc) + (-yL)*xcv - (-yL)*xU, xL*(-ycv) + (-yU)*xcv - (-yU)*xL)


    # # [xm,ym]
    # cv = max(yU*xcv + xU*ycv - xU*yU, yL*xcc + xL*ycc - xL*yL)
    # cc = min(yL*xcv + xU*ycc - xU*yL, yU*xcc + xL*ycv - xL*yU)

end

function transform_rule(::McCormickTransform, ::typeof(/), zL, zU, zcv, zcc, xL, xU, xcv, xcc, yL, yU, ycv, ycc)
    # For division, we do x*(y^-1). Note that if yL < 0 < yU, the inverses of ycv and ycc will
    # be NaN, which will set zcv and zcc to NaN in every case

    # First we calculate the inverse of y
    yL_inv = inv(yU)
    yU_inv = inv(yL)
    ycv_inv = IfElse.ifelse(yL > 0.0, IfElse.ifelse(yU <= ycv, 1.0 ./ ycv, 
            IfElse.ifelse(yU >= ycc, 1.0 ./ ycc, 1.0 ./ yU)),
        IfElse.ifelse(yU < 0.0, IfElse.ifelse(yL == yU, mid_expr(ycc, ycv, yL).^(-1), 
            ((yL.^(-1))*(yU - mid_expr(ycc, ycv, yL)) + (yU.^(-1))*(mid_expr(ycc, ycv, yL) - yL))./(yU - yL)),
                NaN))
    ycc_inv = IfElse.ifelse(yL > 0.0, IfElse.ifelse(yL <= ycv, (yU + yL - ycv)./(yL*yU), 
                IfElse.ifelse(yL >= ycc, (yU + yL - ycc)./(yL*yU), 1.0 ./ yL)),
            IfElse.ifelse(yU < 0.0, mid_expr(ycc, ycv, yU).^(-1),
                NaN))
         
    # Now we use the multiplication rules, but replacing each instance of
    # y with its inverse. 
    rcv = Equation(zcv, IfElse.ifelse(xL >= 0.0,
        IfElse.ifelse(yL_inv >= 0.0, max(zL, max(yU_inv*xcv + xU*ycv_inv - xU*yU_inv, yL_inv*xcv + xL*ycv_inv - xL*yL_inv)),
            IfElse.ifelse(yU_inv <= 0.0, max(zL, -min((-yU_inv)*xcc + xU*(-ycv_inv) - xU*(-yU_inv), (-yL_inv)*xcc + xL*(-ycv_inv) - xL*(-yL_inv))),
                NaN)), #max(zL, max(yU_inv*xcv + xU*ycv_inv - xU*yU_inv, yL_inv*xcc + xL*ycv_inv - xL*yL_inv)))), y must be positive or negative
        IfElse.ifelse(xU <= 0.0,
            IfElse.ifelse(yL_inv >= 0.0, max(zL, -min(yL_inv*(-xcv) + (-xL)*ycc_inv - (-xL)*yL_inv, yU_inv*(-xcv) + (-xU)*ycc_inv - (-xU)*yU_inv)),
                IfElse.ifelse(yU_inv <= 0.0, max(zL, max(yL_inv*xcc + xL*ycc_inv - xL*yL_inv, yU_inv*xcc + xU*ycc_inv - xU*yU_inv)),
                    NaN)), #max(zL, -min(yL_inv*(-xcc) + (-xL)*ycc_inv - (-xL)*yL_inv, yU_inv*(-xcv) + (-xU)*ycc_inv - (-xU)*yU_inv)))), y must be positive or negative
            IfElse.ifelse(yL_inv >= 0.0, max(zL, max(xU*ycv_inv + yU_inv*xcv - yU_inv*xU, xL*ycc_inv + yL_inv*xcv - yL_inv*xL)),
                IfElse.ifelse(yU_inv <= 0.0, max(zL, -min(xL*(-ycc_inv) + (-yL_inv)*xcc - (-yL_inv)*xL, xU*(-ycv_inv) + (-yU_inv)*xcc - (-yU_inv)*xU)), 
                    NaN))))) #max(zL, max(yU_inv*xcv + xU*ycv_inv - xU*yU_inv, yL_inv*xcc + xL*ycc_inv - xL*yL_inv))))))) y must be positive or negative
    rcc = Equation(zcc, IfElse.ifelse(xL >= 0.0,
        IfElse.ifelse(yL_inv >= 0.0, min(zU, min(yL_inv*xcc + xU*ycc_inv - xU*yL_inv, yU_inv*xcc + xL*ycc_inv - xL*yU_inv)),
            IfElse.ifelse(yU_inv <= 0.0, min(zU, -max((-yL_inv)*xcv + xU*(-ycc_inv) - xU*(-yL_inv), (-yU_inv)*xcv + xL*(-ycc_inv) - xL*(-yU_inv))),
                NaN)), #min(zU, min(yL_inv*xcv + xU*ycc_inv - xU*yL_inv, yU_inv*xcc + xL*ycc_inv - xL*yU_inv)))), y must be positive or negative
        IfElse.ifelse(xU <= 0.0,
            IfElse.ifelse(yL_inv >= 0.0, min(zU, -max(yU_inv*(-xcc) + (-xL)*ycv_inv - (-xL)*yU_inv, yL_inv*(-xcc) + (-xU)*ycv_inv - (-xU)*yL_inv)),
                IfElse.ifelse(yU_inv <= 0.0, min(zU, min(yU_inv*xcv + xL*ycv_inv - xL*yU_inv, yL_inv*xcv + xU*ycv_inv - xU*yL_inv)),
                    NaN)), #min(zU, -max(yU_inv*(-xcc) + (-xL)*ycv_inv - (-xL)*yU_inv, yL_inv*(-xcv) + (-xU)*ycv_inv - (-xU)*yL_inv)))), y must be positive or negative
            IfElse.ifelse(yL_inv >= 0.0, min(zU, min(xL*ycv_inv + yU_inv*xcc - yU_inv*xL, xU*ycc_inv + yL_inv*xcc - yL_inv*xU)),
                IfElse.ifelse(yU_inv <= 0.0, min(zU, -max(xU*(-ycc_inv) + (-yL_inv)*xcv - (-yL_inv)*xU, xL*(-ycv_inv) + (-yU_inv)*xcv - (-yU_inv)*xL)), 
                    NaN))))) #min(zU, min(yL_inv*xcv + xU*ycc_inv - xU*yL_inv, yU_inv*xcc + xL*ycv_inv - xL*yU_inv))))))) y must be positive or negative
    return rcv, rcc
end

function transform_rule(::McCormickTransform, ::typeof(min), zL, zU, zcv, zcc, xL, xU, xcv, xcc, yL, yU, ycv, ycc)
    rcv = Equation(zcv, IfElse.ifelse(xU <= yL, xcv, IfElse.ifelse(xL >= yU, ycv, 0.5*(xcv + ycv - abs(xcv - ycv)))))
    rcc = Equation(zcc, min(xcc, ycc))
    return rcv, rcc
end
function transform_rule(::McCormickTransform, ::typeof(min), zL, zU, zcv, zcc, xL, xU, xcv, xcc, yL::Real, yU::Real, ycv::Real, ycc::Real)
    # For min, we do -max(-x, -y). So, replace all instances of x's with -x's, and replace all y's with -y's, then negate the result
    rcv = Equation(zcv, -IfElse.ifelse((-xcv) < (-xcc), IfElse.ifelse((-xcc) < (-xL), IfElse.ifelse((-xL) - (-xU) == 0, max((-xcc), (-ycc)), 
            (max((-xU), (-ycc))*((-xL) - (-xcc)) + max((-xL), (-ycc))*((-xcc) - (-xU)))/((-xL)-(-xU))), IfElse.ifelse((-xL) < (-xcv), 
            IfElse.ifelse((-xL) - (-xU) == 0, max((-xcv), (-ycc)), (max((-xU), (-ycc))*((-xL) - (-xcv)) + max((-xL), (-ycc))*((-xcv) - (-xU)))/((-xL)-(-xU))), 
            IfElse.ifelse((-xL) - (-xU) == 0, max((-xL), (-ycc)), (max((-xL), (-ycc))*((-xL) - (-xU)))/((-xL)-(-xU))))),
            IfElse.ifelse((-xL) < (-xcc), IfElse.ifelse((-xL) - (-xU) == 0, max((-xcc), (-ycc)), 
            (max((-xU), (-ycc))*((-xL) - (-xcc)) + max((-xL), (-ycc))*((-xcc) - (-xU)))/((-xL)-(-xU))), IfElse.ifelse((-xcv) < (-xL), 
            IfElse.ifelse((-xL) - (-xU) == 0, max((-xcv), (-ycc)), (max((-xU), (-ycc))*((-xL) - (-xcv)) + max((-xL), (-ycc))*((-xcv) - (-xU)))/((-xL)-(-xU))), 
            IfElse.ifelse((-xL) - (-xU) == 0, max((-xL), (-ycc)), (max((-xL), (-ycc))*((-xL) - (-xU)))/((-xL)-(-xU)))))))
    rcc = rcc = Equation(zcv, IfElse.ifelse((-xcv) < (-xcc), IfElse.ifelse((-xcc) < (-xU), IfElse.ifelse((-xcc) <= (-ycc), (-ycc), (-xcc)), 
            IfElse.ifelse(c < (-xcv), IfElse.ifelse((-xcv) <= (-ycc), (-ycc), (-xcv)), IfElse.ifelse((-xU) <= (-ycc), (-ycc), (-xU)))),
            IfElse.ifelse((-xU) < (-xcc), IfElse.ifelse((-xcc) <= (-ycc), (-ycc), (-xcc)), IfElse.ifelse((-xcv) < (-xU), IfElse.ifelse((-xcv) <= (-ycc), 
            (-ycc), (-xcv)), IfElse.ifelse((-xU) <= (-ycc), (-ycc), (-xU))))))

    return rcv, rcc
end

function transform_rule(::McCormickTransform, ::typeof(max), zL, zU, zcv, zcc, xL, xU, xcv, xcc, yL, yU, ycv, ycc)
    rcv = Equation(zcv, max(xcv, ycv))
    rcc = Equation(zcc, IfElse.ifelse(xU <= yL, ycc, IfElse.ifelse(xL >= yU, xcc, 0.5*(xcc + ycc + abs(xcc - ycc)))))
    return rcv, rcc
end
function transform_rule(::McCormickTransform, ::typeof(max), zL, zU, zcv, zcc, xL, xU, xcv, xcc, yL::Real, yU::Real, ycv::Real, ycc::Real)
    rcv = Equation(zcv, IfElse.ifelse(xcc < xcv, IfElse.ifelse(xcv < xL, IfElse.ifelse(xcv <= ycv, ycv, xcv), IfElse.ifelse(c < xcc, 
            IfElse.ifelse(xcc <= ycv, ycv, xcc), IfElse.ifelse(xL <= ycv, ycv, xL))), IfElse.ifelse(xL < xcv, IfElse.ifelse(xcv <= ycv, ycv, xcv), 
            IfElse.ifelse(xcc < xL, IfElse.ifelse(xcc <= ycv, ycv, xcc), IfElse.ifelse(xL <= ycv, ycv, xL)))))
    rcc = Equation(zcc, IfElse.ifelse(xcc < xcv, IfElse.ifelse(xcv < xU, IfElse.ifelse(xU - xL == 0, max(xcv, ycc), 
            (max(xL, ycc)*(xU - xcv) + max(xU, ycc)*(xcv - xL))/(xU-xL)), IfElse.ifelse(xU < xcc, IfElse.ifelse(xU - xL == 0, max(xcc, ycc), 
            (max(xL, ycc)*(xU - xcc) + max(xU, ycc)*(xcc - xL))/(xU-xL)), IfElse.ifelse(xU - xL == 0, max(xU, ycc), (max(xU, ycc)*(xU - xL))/(xU-xL)))),
            IfElse.ifelse(xU < xcv, IfElse.ifelse(xU - xL == 0, max(xcv, ycc), (max(xL, ycc)*(xU - xcv) + max(xU, ycc)*(xcv - xL))/(xU-xL)), 
            IfElse.ifelse(xcc < xU, IfElse.ifelse(xU - xL == 0, max(xcc, ycc), (max(xL, ycc)*(xU - xcc) + max(xU, ycc)*(xcc - xL))/(xU-xL)), 
            IfElse.ifelse(xU - xL == 0, max(xU, ycc), (max(xU, ycc)*(xU - xL))/(xU-xL))))))
    return rcv, rcc
end

function transform_rule(::McCormickTransform, ::typeof(^), zL, zU, zcv, zcc, xL, xU, xcv, xcc, yL, yU, ycv, ycc)
    ~((typeof(yL) <: Int) || (typeof(yL) <: AbstractFloat)) && error("Symbolic exponents not currently supported.")
    ~(yL == 2) && error("Exponents besides 2 not currently supported")
    mcv = mid_expr(xcv, xcc, max(min(xU, 0.0), xL))
    mcc = mid_expr(xcv, xcc, IfElse.ifelse(xU < 0.0, xL, IfElse.ifelse(xL > 0, xU, IfElse.ifelse(abs(xL) >= abs(xU), xL, xU))))
    rcv = Equation(zcv, mcv^2)
    rcc = Equation(zcc, (xL+xU)*mcc - xU*xL)
    return rcv, rcc
end


#=
TODO: Add other operators. It's probably helpful to break the McCormick overload and McCormick + Interval Outputs
into separate transform_rules since the coupling for the ODEs are one directional and potentially useful.
=#
