
# This file is specific to gradient calculations and assumes the presence of 
# a matrix of gradients::Matrix{Num}


#=
Unitary Rules
=#  
function grad_transform!(::McCormickIntervalTransform, ::Nothing, zL, zU, zcv, zcc, xL, xU, 
                            xcv, xcc, varlist::Vector{String}, cv_gradlist::Matrix{Num}, cc_gradlist::Matrix{Num})
    z = findfirst(x -> x==string(zL)[1:end-3], varlist)
    x = findfirst(x -> x==string(xL)[1:end-3], varlist)

    # Separate variable check, in case x is a constant
    if isnothing(x)
        x = findfirst(x -> x==string(xL), varlist)
    end

    cv_gradlist[:,z] .= cv_gradlist[:,x]
    cc_gradlist[:,z] .= cc_gradlist[:,x]
    return
end
function grad_transform!(::McCormickIntervalTransform, ::typeof(getindex), zL, zU, zcv, zcc, xL, xU, 
                            xcv, xcc, varlist::Vector{String}, cv_gradlist::Matrix{Num}, cc_gradlist::Matrix{Num})
    z = findfirst(x -> x==string(zL)[1:end-3], varlist)
    x = findfirst(x -> x==string(xL)[1:end-3], varlist)

    # Separate variable check, in case x is a constant
    if isnothing(x)
        x = findfirst(x -> x==string(xL), varlist)
    end

    cv_gradlist[:,z] .= cv_gradlist[:,x]
    cc_gradlist[:,z] .= cc_gradlist[:,x]
    return
end
function grad_transform!(::McCormickIntervalTransform, ::typeof(exp), zL, zU, zcv, zcc, xL, xU, 
                            xcv, xcc, varlist::Vector{String}, cv_gradlist::Matrix{Num}, cc_gradlist::Matrix{Num})
    z = findfirst(x -> x==string(zL)[1:end-3], varlist)
    x = findfirst(x -> x==string(xL)[1:end-3], varlist)

    # Separate variable check, in case x is a constant
    if isnothing(x)
        x = findfirst(x -> x==string(xL), varlist)
    end

    # For cv_grad, we do a mid operation using {xcc, xcv, xL}, and use {exp(xcc), exp(xcv), 0.0} accordingly
    @. cv_gradlist[:,z] = mid_grad(xcc, xcv, xL, cc_gradlist[:,x]*exp(xcc), cv_gradlist[:,x]*exp(xcv), 0.0)

    # For cc_grad, we do the same as above, but we need to check that the interval is non-degenerate first
    @. cc_gradlist[:,z] = IfElse.ifelse(xU - xL == 0.0, 0.0, mid_grad(xcc, xcv, xU, cc_gradlist[:,x]*((exp(xU)-exp(xL))/(xU-xL)), cv_gradlist[:,x]*((exp(xU)-exp(xL))/(xU-xL)), 0.0))
    return
end

#=
Binary Rules
=#
function grad_transform!(::McCormickIntervalTransform, ::typeof(+), zL, zU, zcv, zcc, xL, xU, 
                            xcv, xcc, yL, yU, ycv, ycc, varlist::Vector{String}, cv_gradlist::Matrix{Num}, cc_gradlist::Matrix{Num})
    # Identify which variables are being used
    z = findfirst(x -> x==string(zL)[1:end-3], varlist)
    x = findfirst(x -> x==string(xL)[1:end-3], varlist)
    y = findfirst(x -> x==string(yL)[1:end-3], varlist)

    # Separate variable check, in case x or y are constants
    if isnothing(x)
        x = findfirst(x -> x==string(xL), varlist)
    end
    if isnothing(y)
        y = findfirst(x -> x==string(yL), varlist)
    end

    # Modify the gradlist accordingly
    cv_gradlist[:, z] .= cv_gradlist[:, x] .+ cv_gradlist[:, y]
    cc_gradlist[:, z] .= cc_gradlist[:, x] .+ cc_gradlist[:, y]
    return
end
function grad_transform!(::McCormickIntervalTransform, ::typeof(+), zL, zU, zcv, zcc, xL, xU, 
    xcv, xcc, yL::Real, yU::Real, ycv::Real, ycc::Real, varlist::Vector{String}, cv_gradlist::Matrix{Num}, cc_gradlist::Matrix{Num})
    # Identify which variables are being used
    z = findfirst(x -> x==string(zL)[1:end-3], varlist)
    x = findfirst(x -> x==string(xL)[1:end-3], varlist)

    # Separate variable check, in case x is a constant
    if isnothing(x)
        x = findfirst(x -> x==string(xL), varlist)
    end

    # Modify the gradlist accordingly
    cv_gradlist[:, z] .= cv_gradlist[:, x]
    cc_gradlist[:, z] .= cc_gradlist[:, x]
end
function grad_transform!(::McCormickIntervalTransform, ::typeof(*), zL, zU, zcv, zcc, xL, xU, 
                            xcv, xcc, yL, yU, ycv, ycc, varlist::Vector{String}, cv_gradlist::Matrix{Num}, cc_gradlist::Matrix{Num})
    # Identify which variables are being used
    z = findfirst(x -> x==string(zL)[1:end-3], varlist)
    x = findfirst(x -> x==string(xL)[1:end-3], varlist)
    y = findfirst(x -> x==string(yL)[1:end-3], varlist)

    # Separate variable check, in case x or y are constants
    if isnothing(x)
        x = findfirst(x -> x==string(xL), varlist)
    end
    if isnothing(y)
        y = findfirst(x -> x==string(yL), varlist)
    end

    # Need to include math for {L, U, cv, cc} without the cut operator, so the cut() operator
    # can be used here.
    rL = IfElse.ifelse(yL >= 0.0, 
        IfElse.ifelse(xL >= 0.0, xL*yL,
            IfElse.ifelse(xU <= 0.0, xL*yU, xL*yU)),
        IfElse.ifelse(yU <= 0.0, 
            IfElse.ifelse(xL >= 0.0, xU*yL, 
                IfElse.ifelse(xU <= 0.0, xU*yU, xU*yL)), 
            IfElse.ifelse(xL > 0.0, xU*yL,
                IfElse.ifelse(xU < 0.0, xL*yU, min(xL*yU, xU*yL)))))
    rU = IfElse.ifelse(yL >= 0.0, 
        IfElse.ifelse(xL >= 0.0, xU*yU, 
            IfElse.ifelse(xU <= 0.0, xU*yL, xU*yU)), 
        IfElse.ifelse(yU <= 0.0,
            IfElse.ifelse(xL >= 0.0, xL*yU,
                IfElse.ifelse(xU <= 0.0, xL*yL, xL*yL)),
            IfElse.ifelse(xL > 0.0, xU*yU,
                IfElse.ifelse(xU < 0.0, xL*yL, max(xL*yL, xU*yU))))) 
    rcv = IfElse.ifelse(xL >= 0.0,
        IfElse.ifelse(yL >= 0.0, max(yU*xcv + xU*ycv - xU*yU, yL*xcv + xL*ycv - xL*yL),
            IfElse.ifelse(yU <= 0.0, -min((-yU)*xcc + xU*(-ycv) - xU*(-yU), (-yL)*xcc + xL*(-ycv) - xL*(-yL)),
                max(yU*xcv + xU*ycv - xU*yU, yL*xcc + xL*ycv - xL*yL))),
        IfElse.ifelse(xU <= 0.0,
            IfElse.ifelse(yL >= 0.0, -min(yL*(-xcv) + (-xL)*ycc - (-xL)*yL, yU*(-xcv) + (-xU)*ycc - (-xU)*yU),
                IfElse.ifelse(yU <= 0.0, max(yL*xcc + xL*ycc - xL*yL, yU*xcc + xU*ycc - xU*yU),
                    -min(yL*(-xcc) + (-xL)*ycc - (-xL)*yL, yU*(-xcv) + (-xU)*ycc - (-xU)*yU))),
            IfElse.ifelse(yL >= 0.0, max(xU*ycv + yU*xcv - yU*xU, xL*ycc + yL*xcv - yL*xL),
                IfElse.ifelse(yU <= 0.0, -min(xL*(-ycc) + (-yL)*xcc - (-yL)*xL, xU*(-ycv) + (-yU)*xcc - (-yU)*xU), 
                max(yU*xcv + xU*ycv - xU*yU, yL*xcc + xL*ycc - xL*yL)))))
    rcc = IfElse.ifelse(xL >= 0.0,
        IfElse.ifelse(yL >= 0.0, min(yL*xcc + xU*ycc - xU*yL, yU*xcc + xL*ycc - xL*yU),
            IfElse.ifelse(yU <= 0.0, -max((-yL)*xcv + xU*(-ycc) - xU*(-yL), (-yU)*xcv + xL*(-ycc) - xL*(-yU)),
                min(yL*xcv + xU*ycc - xU*yL, yU*xcc + xL*ycc - xL*yU))),
        IfElse.ifelse(xU <= 0.0,
            IfElse.ifelse(yL >= 0.0, -max(yU*(-xcc) + (-xL)*ycv - (-xL)*yU, yL*(-xcc) + (-xU)*ycv - (-xU)*yL),
                 IfElse.ifelse(yU <= 0.0, min(yU*xcv + xL*ycv - xL*yU, yL*xcv + xU*ycv - xU*yL),
                 -max(yU*(-xcc) + (-xL)*ycv - (-xL)*yU, yL*(-xcv) + (-xU)*ycv - (-xU)*yL))),
            IfElse.ifelse(yL >= 0.0, min(xL*ycv + yU*xcc - yU*xL, xU*ycc + yL*xcc - yL*xU),
                IfElse.ifelse(yU <= 0.0, -max(xU*(-ycc) + (-yL)*xcv - (-yL)*xU, xL*(-ycv) + (-yU)*xcv - (-yU)*xL), 
                    min(yL*xcv + xU*ycc - xU*yL, yU*xcc + xL*ycv - xL*yU)))))

    # Include the cut definition in the gradlist, setting the gradient to the zero vector
    # if cv would have been cut
    zero_vec = Num.(zeros(size(cv_gradlist[:,y])))
    @. cv_gradlist[:, z] = IfElse.ifelse(rcv < rL, zero_vec, IfElse.ifelse(xL >= 0.0, 
        IfElse.ifelse(yL >= 0.0, IfElse.ifelse(yU*xcv + xU*ycv - xU*yU > yL*xcv + xL*ycv - xL*yL, 
                                               yU*cv_gradlist[:,x] + xU*cv_gradlist[:,y], 
                                               yL*cv_gradlist[:,x] + xL*cv_gradlist[:,y]),
            IfElse.ifelse(yU <= 0.0, -IfElse.ifelse((-yU)*xcc + xU*(-ycv) - xU*(-yU) < (-yL)*xcc + xL*(-ycv) - xL*(-yL), 
                                                    (-yU)*cc_gradlist[:,x] + xU*(-cv_gradlist[:,y]),
                                                    (-yL)*cc_gradlist[:,x] + xL*(-cv_gradlist[:,y])),
                IfElse.ifelse(yU*xcv + xU*ycv - xU*yU > yL*xcc + xL*ycv - xL*yL,
                              yU*cv_gradlist[:,x] + xU*cv_gradlist[:,y],
                              yL*cc_gradlist[:,x] + xL*cv_gradlist[:,y]))),
        IfElse.ifelse(xU <= 0.0,
            IfElse.ifelse(yL >= 0.0, -IfElse.ifelse(yL*(-xcv) + (-xL)*ycc - (-xL)*yL < yU*(-xcv) + (-xU)*ycc - (-xU)*yU, 
                                                    yL*(-cv_gradlist[:,x]) + (-xL)*cc_gradlist[:,y],
                                                    yU*(-cv_gradlist[:,x]) + (-xU)*cc_gradlist[:,y]),
                IfElse.ifelse(yU <= 0.0, IfElse.ifelse((-yL)*(-xcc) + (-xL)*(-ycc) - (-xL)*(-yL) > (-yU)*(-xcc) + (-xU)*(-ycc) - (-xU)*(-yU), 
                                                       (-yL)*(-cc_gradlist[:,x]) + (-xL)*(-cc_gradlist[:,y]), 
                                                       (-yU)*(-cc_gradlist[:,x]) + (-xU)*(-cc_gradlist[:,y])), 
                    -IfElse.ifelse(yL*(-xcc) + (-xL)*ycc - (-xL)*yL < yU*(-xcv) + (-xU)*ycc - (-xU)*yU,
                                yL*(-cc_gradlist[:,x]) + (-xL)*cc_gradlist[:,y],
                                yU*(-cv_gradlist[:,x]) + (-xU)*cc_gradlist[:,y]))),
            IfElse.ifelse(yL >= 0.0, IfElse.ifelse((xU)*(ycv) + (yU)*(xcv) - (yU)*(xU) > (xL)*(ycc) + (yL)*(xcv) - (yL)*(xL),
                                                   (xU)*(cv_gradlist[:,y]) + (yU)*(cv_gradlist[:,x]),
                                                   (xL)*(cc_gradlist[:,y]) + (yL)*(cv_gradlist[:,x])), 
                IfElse.ifelse(yU <= 0.0, -IfElse.ifelse((xL)*(-ycc) + (-yL)*(xcc) - (-yL)*(xL) < (xU)*(-ycv) + (-yU)*(xcc) - (-yU)*(xU),
                                                        (xL)*(-cc_gradlist[:,y]) + (-yL)*(cc_gradlist[:,x]),
                                                        (xU)*(-cv_gradlist[:,y]) + (-yU)*(cc_gradlist[:,x])),
                    IfElse.ifelse(yU*xcv + xU*ycv - xU*yU > yL*xcc + xL*ycc - xL*yL, 
                                  yU*cv_gradlist[:,x] + xU*cv_gradlist[:,y], 
                                  yL*cc_gradlist[:,x] + xL*cc_gradlist[:,y]))))))
    @. cc_gradlist[:, z] = IfElse.ifelse(rcc > rU, zero_vec, IfElse.ifelse(xL >= 0.0, 
        IfElse.ifelse(yL >= 0.0, IfElse.ifelse(yL*xcc + xU*ycc - xU*yL < yU*xcc + xL*ycc - xL*yU, 
                                            yL*cc_gradlist[:,x] + xU*cc_gradlist[:,y],
                                            yU*cc_gradlist[:,x] + xL*cc_gradlist[:,y]),
            IfElse.ifelse(yU <= 0.0, -IfElse.ifelse((-yL)*xcv + xU*(-ycc) - xU*(-yL) > (-yU)*xcv + xL*(-ycc) - xL*(-yU), 
                                                    (-yL)*cv_gradlist[:,x] + xU*(-cc_gradlist[:,y]), 
                                                    (-yU)*cv_gradlist[:,x] + xL*(-cc_gradlist[:,y])),
                IfElse.ifelse(yL*xcv + xU*ycc - xU*yL < yU*xcc + xL*ycc - xL*yU,
                                yL*cv_gradlist[:,x] + xU*cc_gradlist[:,y],
                                yU*cc_gradlist[:,x] + xL*cc_gradlist[:,y]))),
        IfElse.ifelse(xU <= 0.0,
            IfElse.ifelse(yL >= 0.0, -IfElse.ifelse(yU*(-xcc) + (-xL)*ycv - (-xL)*yU > yL*(-xcc) + (-xU)*ycv - (-xU)*yL, 
                                                    yU*(-cc_gradlist[:,x]) + (-xL)*cv_gradlist[:,y], 
                                                    yL*(-cc_gradlist[:,x]) + (-xU)*cv_gradlist[:,y]),
                IfElse.ifelse(yU <= 0.0, IfElse.ifelse((-yU)*(-xcv) + (-xL)*(-ycv) - (-xL)*(-yU) < (-yL)*(-xcv) + (-xU)*(-ycv) - (-xU)*(-yL), 
                                                    (-yU)*(-cv_gradlist[:,x]) + (-xL)*(-cv_gradlist[:,y]),
                                                    (-yL)*(-cv_gradlist[:,x]) + (-xU)*(-cv_gradlist[:,y])), 
                    -IfElse.ifelse((-xL)*(ycv) + (yU)*(-xcc) - (yU)*(-xL) > (-xU)*(ycv) + (yL)*(-xcv) - (yL)*(-xU), 
                                   (-xL)*(cv_gradlist[:,y]) + (yU)*(-cc_gradlist[:,x]), 
                                   (-xU)*(cv_gradlist[:,y]) + (yL)*(-cv_gradlist[:,x])))),
            IfElse.ifelse(yL >= 0.0, IfElse.ifelse((xL)*(ycv) + (yU)*(xcc) - (yU)*(xL) < (xU)*(ycc) + (yL)*(xcc) - (yL)*(xU),
                                                (xL)*(cv_gradlist[:,y]) + (yU)*(cc_gradlist[:,x]),
                                                (xU)*(cc_gradlist[:,y]) + (yL)*(cc_gradlist[:,x])), 
                IfElse.ifelse(yU <= 0.0, -IfElse.ifelse((xU)*(-ycc) + (-yL)*(xcv) - (-yL)*(xU) > (xL)*(-ycv) + (-yU)*(xcv) - (-yU)*(xL),
                                                        (xU)*(-cc_gradlist[:,y]) + (-yL)*(cv_gradlist[:,x]),
                                                        (xL)*(-cv_gradlist[:,y]) + (-yU)*(cv_gradlist[:,x])), 
                    IfElse.ifelse(yL*xcv + xU*ycc - xU*yL < yU*xcc + xL*ycv - xL*yU, 
                                yL*cv_gradlist[:,x] + xU*cc_gradlist[:,y],
                                yU*cc_gradlist[:,x] + xL*cv_gradlist[:,y]))))))
    return
end
function grad_transform!(::McCormickIntervalTransform, ::typeof(*), zL, zU, zcv, zcc, xL, xU, 
                            xcv, xcc, yL::Real, yU::Real, ycv::Real, ycc::Real, varlist::Vector{String}, cv_gradlist::Matrix{Num}, cc_gradlist::Matrix{Num})
    # Identify which variables are being used
    z = findfirst(x -> x==string(zL)[1:end-3], varlist)
    x = findfirst(x -> x==string(xL)[1:end-3], varlist)

    # Separate variable check, in case x is a constant
    if isnothing(x)
        x = findfirst(x -> x==string(xL), varlist)
    end

    @. cv_gradlist[:, z] = IfElse.ifelse(xL >= 0.0, 
        IfElse.ifelse(yL >= 0.0, yU*cv_gradlist[:,x],
            IfElse.ifelse(yU <= 0.0, yU*cc_gradlist[:,x],
                IfElse.ifelse(xcv > xcc, yU*cv_gradlist[:,x], yL*cc_gradlist[:,x]))),
        IfElse.ifelse(xU <= 0.0,
            IfElse.ifelse(yL >= 0.0, yL*cv_gradlist[:,x],
                    IfElse.ifelse(yU <= 0.0, yL*cc_gradlist[:,x], yL*cv_gradlist[:,x])), 
                IfElse.ifelse(yL >= 0.0, yU*cv_gradlist[:,x], 
                    IfElse.ifelse(yU <= 0.0, yL*cc_gradlist[:,x],
                        IfElse.ifelse(xcv > xcc, yU*cv_gradlist[:,x], yL*cc_gradlist[:,x])))))
    @. cc_gradlist[:, z] = IfElse.ifelse(xL >= 0.0, 
        IfElse.ifelse(yL >= 0.0, yL*cc_gradlist[:,x],
            IfElse.ifelse(yU <= 0.0, yU*cv_gradlist[:,x],
                IfElse.ifelse(xcv < xcc, yL*cv_gradlist[:,x], yU*cc_gradlist[:,x]))),
        IfElse.ifelse(xU <= 0.0,
            IfElse.ifelse(yL >= 0.0, yL*cc_gradlist[:,x],
                IfElse.ifelse(yU <= 0.0, yU*cv_gradlist[:,x], yU*cc_gradlist[:,x])),
            IfElse.ifelse(yL >= 0.0, yU*cc_gradlist[:,x], 
                IfElse.ifelse(yU <= 0.0, yL*cv_gradlist[:,x], 
                    IfElse.ifelse(xcv < xcc, yL*cv_gradlist[:,x], yU*cc_gradlist[:,x])))))
    return
end
function grad_transform!(::McCormickIntervalTransform, ::typeof(/), zL, zU, zcv, zcc, xL, xU, 
                            xcv, xcc, yL, yU, ycv, ycc, varlist::Vector{String}, cv_gradlist::Matrix{Num}, cc_gradlist::Matrix{Num})
    # Identify which variables are being used
    z = findfirst(x -> x==string(zL)[1:end-3], varlist)
    x = findfirst(x -> x==string(xL)[1:end-3], varlist)
    y = findfirst(x -> x==string(yL)[1:end-3], varlist)

    # Separate variable check, in case x or y are constants
    if isnothing(x)
        x = findfirst(x -> x==string(xL), varlist)
    end
    if isnothing(y)
        y = findfirst(x -> x==string(yL), varlist)
    end

    # For division, we do x*(y^-1). Note that if yL < 0 < yU, the inverses of ycv, ycc, and their subgradients will
    # be NaN, which will set zcv, zcc, and their subgradients to NaN in every case. First, we define the vector
    # of zeros for the subgradient
    zero_vec = Num.(zeros(size(cv_gradlist[:,y])))

    # Next we calculate the inverse of y
    yL_inv = inv(yU)
    yU_inv = inv(yL)
    ycv_inv = IfElse.ifelse(yL > 0.0, 1.0 ./ (mid_expr(ycc, ycv, yU)), 
            IfElse.ifelse(yU < 0.0, IfElse.ifelse(yL == yU, mid_expr(ycc, ycv, yL).^(-1), (yL.^(-1).*(yU - mid_expr(ycc, ycv, yL)) + yU.^(-1).*(mid_expr(ycc, ycv, yL) - yL))./(yU - yL)),
                NaN))
    ycc_inv = IfElse.ifelse(yL > 0.0, (yU + yL - mid_expr(ycc, ycv, yL))./(yL*yU), 
            IfElse.ifelse(yU < 0.0, mid_expr(ycc, ycv, yU).^(-1),
                NaN))
    y_cv_gradlist_inv = similar(cv_gradlist[:,y])
    @. y_cv_gradlist_inv = IfElse.ifelse(yU < 0.0, IfElse.ifelse(yU == yL, -1/(mid_expr(ycc, ycv, yL)*mid_expr(ycc, ycv, yL)) * mid_grad(ycc, ycv, yL, cc_gradlist[:,y], cv_gradlist[:,y], zero_vec), (yU^-1 - yL^-1)/(yU - yL)) * 
                mid_grad(ycc, ycv, yL, cc_gradlist[:,y], cv_gradlist[:,y], zero_vec),
        IfElse.ifelse(yL > 0.0, -1.0/(mid_expr(ycc, ycv, yU)*mid_expr(ycc, ycv, yU)) * 
                mid_grad(ycc, ycv, yU, cc_gradlist[:,y], cv_gradlist[:,y], zero_vec),
        NaN * zero_vec))
    y_cc_gradlist_inv = similar(cc_gradlist[:,y])
    @. y_cc_gradlist_inv = IfElse.ifelse(yU < 0.0, -1/(mid_expr(ycc, ycv, yU)*mid_expr(ycc, ycv, yU)) *
                mid_grad(ycc, ycv, yU, cc_gradlist[:,y], cv_gradlist[:,y], zero_vec),
        IfElse.ifelse(yL > 0.0, -1.0/(yL*yU) * 
                mid_grad(ycc, ycv, yL, cc_gradlist[:,y], cv_gradlist[:,y], zero_vec),
        NaN * zero_vec))

    # Now we use the multiplication rules, but replace each instance of
    # y with its inverse. 


    # Include the cut definition in the gradlist, setting the gradient to the zero vector
    # if cv would have been cut
    # zero_vec = Num.(zeros(size(cv_gradlist[:,y])))

    @. cv_gradlist[:, z] = IfElse.ifelse(xL >= 0.0, 
        IfElse.ifelse(yL_inv >= 0.0, IfElse.ifelse(yU_inv*xcv + xU*ycv_inv - xU*yU_inv > yL_inv*xcv + xL*ycv_inv - xL*yL_inv, 
                                            yU_inv*cv_gradlist[:,x] + xU*(y_cv_gradlist_inv), 
                                            yL_inv*cv_gradlist[:,x] + xL*(y_cv_gradlist_inv)),
            IfElse.ifelse(yU_inv <= 0.0, -IfElse.ifelse((-yU_inv)*xcc + xU*(-ycv_inv) - xU*(-yU_inv) < (-yL_inv)*xcc + xL*(-ycv_inv) - xL*(-yL_inv), 
                                                    (-yU_inv)*cc_gradlist[:,x] + xU*(-(y_cv_gradlist_inv)),
                                                    (-yL_inv)*cc_gradlist[:,x] + xL*(-(y_cv_gradlist_inv))),
                IfElse.ifelse(yU_inv*xcv + xU*ycv_inv - xU*yU_inv > yL_inv*xcc + xL*ycv_inv - xL*yL_inv,
                            yU_inv*cv_gradlist[:,x] + xU*(y_cv_gradlist_inv),
                            yL_inv*cc_gradlist[:,x] + xL*(y_cv_gradlist_inv)))),
        IfElse.ifelse(xU <= 0.0,
            IfElse.ifelse(yL_inv >= 0.0, -IfElse.ifelse(yL_inv*(-xcv) + (-xL)*ycc_inv - (-xL)*yL_inv < yU_inv*(-xcv) + (-xU)*ycc_inv - (-xU)*yU_inv, 
                                                    yL_inv*(-cv_gradlist[:,x]) + (-xL)*(y_cc_gradlist_inv),
                                                    yU_inv*(-cv_gradlist[:,x]) + (-xU)*(y_cc_gradlist_inv)),
                IfElse.ifelse(yU_inv <= 0.0, IfElse.ifelse((-yL_inv)*(-xcc) + (-xL)*(-ycc_inv) - (-xL)*(-yL_inv) > (-yU_inv)*(-xcc) + (-xU)*(-ycc_inv) - (-xU)*(-yU_inv), 
                                                    (-yL_inv)*(-cc_gradlist[:,x]) + (-xL)*(-(y_cc_gradlist_inv)), 
                                                    (-yU_inv)*(-cc_gradlist[:,x]) + (-xU)*(-(y_cc_gradlist_inv))), 
                    -IfElse.ifelse((-xU)*(ycc_inv) + (yU_inv)*(-xcv) - (yU_inv)*(-xU) < (-xL)*(ycc_inv) + (yL_inv)*(-xcv) - (yL_inv)*(-xL), 
                                (-xU)*((y_cc_gradlist_inv)) + (yU_inv)*(-cv_gradlist[:,x]),
                                (-xL)*((y_cc_gradlist_inv)) + (yL_inv)*(-cv_gradlist[:,x])))),
            IfElse.ifelse(yL_inv >= 0.0, IfElse.ifelse((xU)*(ycv_inv) + (yU_inv)*(xcv) - (yU_inv)*(xU) > (xL)*(ycc_inv) + (yL_inv)*(xcv) - (yL_inv)*(xL),
                                                (xU)*((y_cv_gradlist_inv)) + (yU_inv)*(cv_gradlist[:,x]),
                                                (xL)*((y_cc_gradlist_inv)) + (yL_inv)*(cv_gradlist[:,x])), 
                IfElse.ifelse(yU_inv <= 0.0, -IfElse.ifelse((xL)*(-ycc_inv) + (-yL_inv)*(xcc) - (-yL_inv)*(xL) < (xU)*(-ycv_inv) + (-yU_inv)*(xcc) - (-yU_inv)*(xU),
                                                        (xL)*(-(y_cc_gradlist_inv)) + (-yL_inv)*(cc_gradlist[:,x]),
                                                        (xU)*(-(y_cv_gradlist_inv)) + (-yU_inv)*(cc_gradlist[:,x])),
                    IfElse.ifelse(yU_inv*xcv + xU*ycv_inv - xU*yU_inv > yL_inv*xcc + xL*ycc_inv - xL*yL_inv, 
                                yU_inv*cv_gradlist[:,x] + xU*(y_cv_gradlist_inv), 
                                yL_inv*cc_gradlist[:,x] + xL*(y_cc_gradlist_inv))))))

    @. cc_gradlist[:, z] = IfElse.ifelse(xL >= 0.0, 
        IfElse.ifelse(yL_inv >= 0.0, IfElse.ifelse(yL_inv*xcc + xU*ycc_inv - xU*yL_inv < yU_inv*xcc + xL*ycc_inv - xL*yU_inv, 
                                            yL_inv*cc_gradlist[:,x] + xU*(y_cc_gradlist_inv),
                                            yU_inv*cc_gradlist[:,x] + xL*(y_cc_gradlist_inv)),
            IfElse.ifelse(yU_inv <= 0.0, -IfElse.ifelse((-yL_inv)*xcv + xU*(-ycc_inv) - xU*(-yL_inv) > (-yU_inv)*xcv + xL*(-ycc_inv) - xL*(-yU_inv), 
                                                    (-yL_inv)*cv_gradlist[:,x] + xU*(-(y_cc_gradlist_inv)), 
                                                    (-yU_inv)*cv_gradlist[:,x] + xL*(-(y_cc_gradlist_inv))),
                IfElse.ifelse(yL_inv*xcv + xU*ycc_inv - xU*yL_inv < yU_inv*xcc + xL*ycc_inv - xL*yU_inv,
                                yL_inv*cv_gradlist[:,x] + xU*(y_cc_gradlist_inv),
                                yU_inv*cc_gradlist[:,x] + xL*(y_cc_gradlist_inv)))),
        IfElse.ifelse(xU <= 0.0,
            IfElse.ifelse(yL_inv >= 0.0, -IfElse.ifelse(yU_inv*(-xcc) + (-xL)*ycv_inv - (-xL)*yU_inv > yL_inv*(-xcc) + (-xU)*ycv_inv - (-xU)*yL_inv, 
                                                    yU_inv*(-cc_gradlist[:,x]) + (-xL)*(y_cv_gradlist_inv), 
                                                    yL_inv*(-cc_gradlist[:,x]) + (-xU)*(y_cv_gradlist_inv)),
                IfElse.ifelse(yU_inv <= 0.0, IfElse.ifelse((-yU_inv)*(-xcv) + (-xL)*(-ycv_inv) - (-xL)*(-yU_inv) < (-yL_inv)*(-xcv) + (-xU)*(-ycv_inv) - (-xU)*(-yL_inv), 
                                                    (-yU_inv)*(-cv_gradlist[:,x]) + (-xL)*(-(y_cv_gradlist_inv)),
                                                    (-yL_inv)*(-cv_gradlist[:,x]) + (-xU)*(-(y_cv_gradlist_inv))), 
                    -IfElse.ifelse((-xL)*(ycv_inv) + (yU_inv)*(-xcc) - (yU_inv)*(-xL) > (-xU)*(ycv_inv) + (yL_inv)*(-xcc) - (yL_inv)*(-xU), 
                                (-xL)*((y_cv_gradlist_inv)) + (yU_inv)*(-cc_gradlist[:,x]), 
                                (-xU)*((y_cv_gradlist_inv)) + (yL_inv)*(-cc_gradlist[:,x])))),
            IfElse.ifelse(yL_inv >= 0.0, IfElse.ifelse((xL)*(ycv_inv) + (yU_inv)*(xcc) - (yU_inv)*(xL) < (xU)*(ycc_inv) + (yL_inv)*(xcc) - (yL_inv)*(xU),
                                                (xL)*((y_cv_gradlist_inv)) + (yU_inv)*(cc_gradlist[:,x]),
                                                (xU)*((y_cc_gradlist_inv)) + (yL_inv)*(cc_gradlist[:,x])), 
                IfElse.ifelse(yU_inv <= 0.0, -IfElse.ifelse((xU)*(-ycc_inv) + (-yL_inv)*(xcv) - (-yL_inv)*(xU) > (xL)*(-ycv_inv) + (-yU_inv)*(xcv) - (-yU_inv)*(xL),
                                                        (xU)*(-(y_cc_gradlist_inv)) + (-yL_inv)*(cv_gradlist[:,x]),
                                                        (xL)*(-(y_cv_gradlist_inv)) + (-yU_inv)*(cv_gradlist[:,x])), 
                    IfElse.ifelse(yL_inv*xcv + xU*ycc_inv - xU*yL_inv < yU_inv*xcc + xL*ycv_inv - xL*yU_inv, 
                                yL_inv*cv_gradlist[:,x] + xU*(y_cc_gradlist_inv),
                                yU_inv*cc_gradlist[:,x] + xL*(y_cv_gradlist_inv))))))
    return
end
function grad_transform!(::McCormickIntervalTransform, ::typeof(/), zL, zU, zcv, zcc, xL::Real, xU::Real, 
                            xcv::Real, xcc::Real, yL, yU, ycv, ycc, varlist::Vector{String}, cv_gradlist::Matrix{Num}, cc_gradlist::Matrix{Num})
    # Identify which variables are being used
    z = findfirst(x -> x==string(zL)[1:end-3], varlist)
    y = findfirst(x -> x==string(yL)[1:end-3], varlist)

    # Separate variable check, in case y is a constant
    if isnothing(y)
        x = findfirst(x -> x==string(yL), varlist)
    end

    # For division, we do x*(y^-1). Note that if yL < 0 < yU, the inverses of ycv, ycc, and their subgradients will
    # be NaN, which will set zcv, zcc, and their subgradients to NaN in every case. First, we define the vector
    # of zeros for the subgradient
    zero_vec = Num.(zeros(size(cv_gradlist[:,y])))

    # Next we calculate the inverse of y
    yL_inv = inv(yU)
    yU_inv = inv(yL)
    ycv_inv = IfElse.ifelse(yL > 0.0, 1.0 ./ (mid_expr(ycc, ycv, yU)), 
            IfElse.ifelse(yU < 0.0, IfElse.ifelse(yL == yU, mid_expr(ycc, ycv, yL).^(-1), (yL.^(-1).*(yU - mid_expr(ycc, ycv, yL)) + yU.^(-1).*(mid_expr(ycc, ycv, yL) - yL))./(yU - yL)),
                NaN))
    ycc_inv = IfElse.ifelse(yL > 0.0, (yU + yL - mid_expr(ycc, ycv, yL))./(yL*yU), 
            IfElse.ifelse(yU < 0.0, mid_expr(ycc, ycv, yU).^(-1),
                NaN))
    y_cv_gradlist_inv = similar(cv_gradlist[:,y])
    @. y_cv_gradlist_inv = IfElse.ifelse(yU < 0.0, IfElse.ifelse(yU == yL, -1/(mid_expr(ycc, ycv, yL)*mid_expr(ycc, ycv, yL)), (yU^-1 - yL^-1)/(yU - yL)) * 
                mid_grad(ycc, ycv, yL, cc_gradlist[:,y], cv_gradlist[:,y], zero_vec),
        IfElse.ifelse(yL > 0.0, -1.0/(mid_expr(ycc, ycv, yU)*mid_expr(ycc, ycv, yU)) * 
                mid_grad(ycc, ycv, yU, cc_gradlist[:,y], cv_gradlist[:,y], zero_vec),
        NaN * zero_vec))
    y_cc_gradlist_inv = similar(cc_gradlist[:,y])
    @. y_cc_gradlist_inv = IfElse.ifelse(yU < 0.0, -1/(mid_expr(ycc, ycv, yU)*mid_expr(ycc, ycv, yU)) *
                mid_grad(ycc, ycv, yU, cc_gradlist[:,y], cv_gradlist[:,y], zero_vec),
        IfElse.ifelse(yL > 0.0, -1.0/(yL*yU) * 
                mid_grad(ycc, ycv, yL, cc_gradlist[:,y], cv_gradlist[:,y], zero_vec),
        NaN * zero_vec))

    # Now we use the multiplication rules, but replace each instance of
    # y with its inverse. 
    @. cv_gradlist[:, z] = IfElse.ifelse(xL >= 0.0, 
        IfElse.ifelse(yL_inv >= 0.0, IfElse.ifelse(yU_inv*xcv + xU*ycv_inv - xU*yU_inv > yL_inv*xcv + xL*ycv_inv - xL*yL_inv, 
                                            xU*(y_cv_gradlist_inv), 
                                            xL*(y_cv_gradlist_inv)),
            IfElse.ifelse(yU_inv <= 0.0, -IfElse.ifelse((-yU_inv)*xcc + xU*(-ycv_inv) - xU*(-yU_inv) < (-yL_inv)*xcc + xL*(-ycv_inv) - xL*(-yL_inv), 
                                                    xU*(-(y_cv_gradlist_inv)),
                                                    xL*(-(y_cv_gradlist_inv))),
                IfElse.ifelse(yU_inv*xcv + xU*ycv_inv - xU*yU_inv > yL_inv*xcc + xL*ycv_inv - xL*yL_inv,
                            xU*(y_cv_gradlist_inv),
                            xL*(y_cv_gradlist_inv)))),
        IfElse.ifelse(xU <= 0.0,
            IfElse.ifelse(yL_inv >= 0.0, -IfElse.ifelse(yL_inv*(-xcv) + (-xL)*ycc_inv - (-xL)*yL_inv < yU_inv*(-xcv) + (-xU)*ycc_inv - (-xU)*yU_inv, 
                                                    (-xL)*(y_cc_gradlist_inv),
                                                    (-xU)*(y_cc_gradlist_inv)),
                IfElse.ifelse(yU_inv <= 0.0, IfElse.ifelse((-yL_inv)*(-xcc) + (-xL)*(-ycc_inv) - (-xL)*(-yL_inv) > (-yU_inv)*(-xcc) + (-xU)*(-ycc_inv) - (-xU)*(-yU_inv), 
                                                    (-xL)*(-(y_cc_gradlist_inv)), 
                                                    (-xU)*(-(y_cc_gradlist_inv))), 
                    -IfElse.ifelse((-xU)*(ycc_inv) + (yU_inv)*(-xcv) - (yU_inv)*(-xU) < (-xL)*(ycc_inv) + (yL_inv)*(-xcv) - (yL_inv)*(-xL), 
                                (-xU)*((y_cc_gradlist_inv)),
                                (-xL)*((y_cc_gradlist_inv))))),
            IfElse.ifelse(yL_inv >= 0.0, IfElse.ifelse((xU)*(ycv_inv) + (yU_inv)*(xcv) - (yU_inv)*(xU) > (xL)*(ycc_inv) + (yL_inv)*(xcv) - (yL_inv)*(xL),
                                                (xU)*((y_cv_gradlist_inv)),
                                                (xL)*((y_cc_gradlist_inv))), 
                IfElse.ifelse(yU_inv <= 0.0, -IfElse.ifelse((xL)*(-ycc_inv) + (-yL_inv)*(xcc) - (-yL_inv)*(xL) < (xU)*(-ycv_inv) + (-yU_inv)*(xcc) - (-yU_inv)*(xU),
                                                        (xL)*(-(y_cc_gradlist_inv)),
                                                        (xU)*(-(y_cv_gradlist_inv))),
                    IfElse.ifelse(yU_inv*xcv + xU*ycv_inv - xU*yU_inv > yL_inv*xcc + xL*ycc_inv - xL*yL_inv, 
                                xU*(y_cv_gradlist_inv), 
                                xL*(y_cc_gradlist_inv))))))

    @. cc_gradlist[:, z] = IfElse.ifelse(xL >= 0.0, 
        IfElse.ifelse(yL_inv >= 0.0, IfElse.ifelse(yL_inv*xcc + xU*ycc_inv - xU*yL_inv < yU_inv*xcc + xL*ycc_inv - xL*yU_inv, 
                                            xU*(y_cc_gradlist_inv),
                                            xL*(y_cc_gradlist_inv)),
            IfElse.ifelse(yU_inv <= 0.0, -IfElse.ifelse((-yL_inv)*xcv + xU*(-ycc_inv) - xU*(-yL_inv) > (-yU_inv)*xcv + xL*(-ycc_inv) - xL*(-yU_inv), 
                                                    xU*(-(y_cc_gradlist_inv)), 
                                                    xL*(-(y_cc_gradlist_inv))),
                IfElse.ifelse(yL_inv*xcv + xU*ycc_inv - xU*yL_inv < yU_inv*xcc + xL*ycc_inv - xL*yU_inv,
                                xU*(y_cc_gradlist_inv),
                                xL*(y_cc_gradlist_inv)))),
        IfElse.ifelse(xU <= 0.0,
            IfElse.ifelse(yL_inv >= 0.0, -IfElse.ifelse(yU_inv*(-xcc) + (-xL)*ycv_inv - (-xL)*yU_inv > yL_inv*(-xcc) + (-xU)*ycv_inv - (-xU)*yL_inv, 
                                                    (-xL)*(y_cv_gradlist_inv), 
                                                    (-xU)*(y_cv_gradlist_inv)),
                IfElse.ifelse(yU_inv <= 0.0, IfElse.ifelse((-yU_inv)*(-xcv) + (-xL)*(-ycv_inv) - (-xL)*(-yU_inv) < (-yL_inv)*(-xcv) + (-xU)*(-ycv_inv) - (-xU)*(-yL_inv), 
                                                    (-xL)*(-(y_cv_gradlist_inv)),
                                                    (-xU)*(-(y_cv_gradlist_inv))), 
                    -IfElse.ifelse((-xL)*(ycv_inv) + (yU_inv)*(-xcc) - (yU_inv)*(-xL) > (-xU)*(ycv_inv) + (yL_inv)*(-xcc) - (yL_inv)*(-xU), 
                                (-xL)*((y_cv_gradlist_inv)), 
                                (-xU)*((y_cv_gradlist_inv))))),
            IfElse.ifelse(yL_inv >= 0.0, IfElse.ifelse((xL)*(ycv_inv) + (yU_inv)*(xcc) - (yU_inv)*(xL) < (xU)*(ycc_inv) + (yL_inv)*(xcc) - (yL_inv)*(xU),
                                                (xL)*((y_cv_gradlist_inv)),
                                                (xU)*((y_cc_gradlist_inv))), 
                IfElse.ifelse(yU_inv <= 0.0, -IfElse.ifelse((xU)*(-ycc_inv) + (-yL_inv)*(xcv) - (-yL_inv)*(xU) > (xL)*(-ycv_inv) + (-yU_inv)*(xcv) - (-yU_inv)*(xL),
                                                        (xU)*(-(y_cc_gradlist_inv)),
                                                        (xL)*(-(y_cv_gradlist_inv))), 
                    IfElse.ifelse(yL_inv*xcv + xU*ycc_inv - xU*yL_inv < yU_inv*xcc + xL*ycv_inv - xL*yU_inv, 
                                xU*(y_cc_gradlist_inv),
                                xL*(y_cv_gradlist_inv))))))
    return
end
function grad_transform!(::McCormickIntervalTransform, ::typeof(/), zL, zU, zcv, zcc, xL, xU, 
                            xcv, xcc, yL::Real, yU::Real, ycv::Real, ycc::Real, varlist::Vector{String}, cv_gradlist::Matrix{Num}, cc_gradlist::Matrix{Num})
    # Identify which variables are being used
    z = findfirst(x -> x==string(zL)[1:end-3], varlist)
    x = findfirst(x -> x==string(xL)[1:end-3], varlist)

    # Separate variable check, in case x is a constant
    if isnothing(x)
        x = findfirst(x -> x==string(xL), varlist)
    end

    # For division, we do x*(y^-1). Note that if yL < 0 < yU, the inverses of ycv, ycc, and their subgradients will
    # be NaN, which will set zcv, zcc, and their subgradients to NaN in every case. First, we define the vector
    # of zeros for the subgradient
    zero_vec = Num.(zeros(size(cv_gradlist[:,y])))

    # Next we calculate the inverse of y
    yL_inv = inv(yU)
    yU_inv = inv(yL)
    ycv_inv = inv(ycc)
    ycc_inv = inv(ycv)
    y_cv_gradlist_inv = zero_vec
    y_cc_gradlist_inv = zero_vec

    # Now we use the multiplication rules, but replace each instance of
    # y with its inverse. 
    @. cv_gradlist[:, z] = IfElse.ifelse(xL >= 0.0, 
        IfElse.ifelse(yL_inv >= 0.0, IfElse.ifelse(yU_inv*xcv + xU*ycv_inv - xU*yU_inv > yL_inv*xcv + xL*ycv_inv - xL*yL_inv, 
                                            yU_inv*cv_gradlist[:,x] + xU*(y_cv_gradlist_inv), 
                                            yL_inv*cv_gradlist[:,x] + xL*(y_cv_gradlist_inv)),
            IfElse.ifelse(yU_inv <= 0.0, -IfElse.ifelse((-yU_inv)*xcc + xU*(-ycv_inv) - xU*(-yU_inv) < (-yL_inv)*xcc + xL*(-ycv_inv) - xL*(-yL_inv), 
                                                    (-yU_inv)*cc_gradlist[:,x] + xU*(-(y_cv_gradlist_inv)),
                                                    (-yL_inv)*cc_gradlist[:,x] + xL*(-(y_cv_gradlist_inv))),
                IfElse.ifelse(yU_inv*xcv + xU*ycv_inv - xU*yU_inv > yL_inv*xcc + xL*ycv_inv - xL*yL_inv,
                            yU_inv*cv_gradlist[:,x] + xU*(y_cv_gradlist_inv),
                            yL_inv*cc_gradlist[:,x] + xL*(y_cv_gradlist_inv)))),
        IfElse.ifelse(xU <= 0.0,
            IfElse.ifelse(yL_inv >= 0.0, -IfElse.ifelse(yL_inv*(-xcv) + (-xL)*ycc_inv - (-xL)*yL_inv < yU_inv*(-xcv) + (-xU)*ycc_inv - (-xU)*yU_inv, 
                                                    yL_inv*(-cv_gradlist[:,x]) + (-xL)*(y_cc_gradlist_inv),
                                                    yU_inv*(-cv_gradlist[:,x]) + (-xU)*(y_cc_gradlist_inv)),
                IfElse.ifelse(yU_inv <= 0.0, IfElse.ifelse((-yL_inv)*(-xcc) + (-xL)*(-ycc_inv) - (-xL)*(-yL_inv) > (-yU_inv)*(-xcc) + (-xU)*(-ycc_inv) - (-xU)*(-yU_inv), 
                                                    (-yL_inv)*(-cc_gradlist[:,x]) + (-xL)*(-(y_cc_gradlist_inv)), 
                                                    (-yU_inv)*(-cc_gradlist[:,x]) + (-xU)*(-(y_cc_gradlist_inv))), 
                    -IfElse.ifelse((-xU)*(ycc_inv) + (yU_inv)*(-xcv) - (yU_inv)*(-xU) < (-xL)*(ycc_inv) + (yL_inv)*(-xcv) - (yL_inv)*(-xL), 
                                (-xU)*((y_cc_gradlist_inv)) + (yU_inv)*(-cv_gradlist[:,x]),
                                (-xL)*((y_cc_gradlist_inv)) + (yL_inv)*(-cv_gradlist[:,x])))),
            IfElse.ifelse(yL_inv >= 0.0, IfElse.ifelse((xU)*(ycv_inv) + (yU_inv)*(xcv) - (yU_inv)*(xU) > (xL)*(ycc_inv) + (yL_inv)*(xcv) - (yL_inv)*(xL),
                                                (xU)*((y_cv_gradlist_inv)) + (yU_inv)*(cv_gradlist[:,x]),
                                                (xL)*((y_cc_gradlist_inv)) + (yL_inv)*(cv_gradlist[:,x])), 
                IfElse.ifelse(yU_inv <= 0.0, -IfElse.ifelse((xL)*(-ycc_inv) + (-yL_inv)*(xcc) - (-yL_inv)*(xL) < (xU)*(-ycv_inv) + (-yU_inv)*(xcc) - (-yU_inv)*(xU),
                                                        (xL)*(-(y_cc_gradlist_inv)) + (-yL_inv)*(cc_gradlist[:,x]),
                                                        (xU)*(-(y_cv_gradlist_inv)) + (-yU_inv)*(cc_gradlist[:,x])),
                    IfElse.ifelse(yU_inv*xcv + xU*ycv_inv - xU*yU_inv > yL_inv*xcc + xL*ycc_inv - xL*yL_inv, 
                                yU_inv*cv_gradlist[:,x] + xU*(y_cv_gradlist_inv), 
                                yL_inv*cc_gradlist[:,x] + xL*(y_cc_gradlist_inv))))))

    @. cc_gradlist[:, z] = IfElse.ifelse(xL >= 0.0, 
        IfElse.ifelse(yL_inv >= 0.0, IfElse.ifelse(yL_inv*xcc + xU*ycc_inv - xU*yL_inv < yU_inv*xcc + xL*ycc_inv - xL*yU_inv, 
                                            yL_inv*cc_gradlist[:,x] + xU*(y_cc_gradlist_inv),
                                            yU_inv*cc_gradlist[:,x] + xL*(y_cc_gradlist_inv)),
            IfElse.ifelse(yU_inv <= 0.0, -IfElse.ifelse((-yL_inv)*xcv + xU*(-ycc_inv) - xU*(-yL_inv) > (-yU_inv)*xcv + xL*(-ycc_inv) - xL*(-yU_inv), 
                                                    (-yL_inv)*cv_gradlist[:,x] + xU*(-(y_cc_gradlist_inv)), 
                                                    (-yU_inv)*cv_gradlist[:,x] + xL*(-(y_cc_gradlist_inv))),
                IfElse.ifelse(yL_inv*xcv + xU*ycc_inv - xU*yL_inv < yU_inv*xcc + xL*ycc_inv - xL*yU_inv,
                                yL_inv*cv_gradlist[:,x] + xU*(y_cc_gradlist_inv),
                                yU_inv*cc_gradlist[:,x] + xL*(y_cc_gradlist_inv)))),
        IfElse.ifelse(xU <= 0.0,
            IfElse.ifelse(yL_inv >= 0.0, -IfElse.ifelse(yU_inv*(-xcc) + (-xL)*ycv_inv - (-xL)*yU_inv > yL_inv*(-xcc) + (-xU)*ycv_inv - (-xU)*yL_inv, 
                                                    yU_inv*(-cc_gradlist[:,x]) + (-xL)*(y_cv_gradlist_inv), 
                                                    yL_inv*(-cc_gradlist[:,x]) + (-xU)*(y_cv_gradlist_inv)),
                IfElse.ifelse(yU_inv <= 0.0, IfElse.ifelse((-yU_inv)*(-xcv) + (-xL)*(-ycv_inv) - (-xL)*(-yU_inv) < (-yL_inv)*(-xcv) + (-xU)*(-ycv_inv) - (-xU)*(-yL_inv), 
                                                    (-yU_inv)*(-cv_gradlist[:,x]) + (-xL)*(-(y_cv_gradlist_inv)),
                                                    (-yL_inv)*(-cv_gradlist[:,x]) + (-xU)*(-(y_cv_gradlist_inv))), 
                    -IfElse.ifelse((-xL)*(ycv_inv) + (yU_inv)*(-xcc) - (yU_inv)*(-xL) > (-xU)*(ycv_inv) + (yL_inv)*(-xcc) - (yL_inv)*(-xU), 
                                (-xL)*((y_cv_gradlist_inv)) + (yU_inv)*(-cc_gradlist[:,x]), 
                                (-xU)*((y_cv_gradlist_inv)) + (yL_inv)*(-cc_gradlist[:,x])))),
            IfElse.ifelse(yL_inv >= 0.0, IfElse.ifelse((xL)*(ycv_inv) + (yU_inv)*(xcc) - (yU_inv)*(xL) < (xU)*(ycc_inv) + (yL_inv)*(xcc) - (yL_inv)*(xU),
                                                (xL)*((y_cv_gradlist_inv)) + (yU_inv)*(cc_gradlist[:,x]),
                                                (xU)*((y_cc_gradlist_inv)) + (yL_inv)*(cc_gradlist[:,x])), 
                IfElse.ifelse(yU_inv <= 0.0, -IfElse.ifelse((xU)*(-ycc_inv) + (-yL_inv)*(xcv) - (-yL_inv)*(xU) > (xL)*(-ycv_inv) + (-yU_inv)*(xcv) - (-yU_inv)*(xL),
                                                        (xU)*(-(y_cc_gradlist_inv)) + (-yL_inv)*(cv_gradlist[:,x]),
                                                        (xL)*(-(y_cv_gradlist_inv)) + (-yU_inv)*(cv_gradlist[:,x])), 
                    IfElse.ifelse(yL_inv*xcv + xU*ycc_inv - xU*yL_inv < yU_inv*xcc + xL*ycv_inv - xL*yU_inv, 
                                yL_inv*cv_gradlist[:,x] + xU*(y_cc_gradlist_inv),
                                yU_inv*cc_gradlist[:,x] + xL*(y_cv_gradlist_inv))))))
    return
end
function grad_transform!(::McCormickIntervalTransform, ::typeof(^), zL, zU, zcv, zcc, xL, xU, 
                            xcv, xcc, yL, yU, ycv, ycc, varlist::Vector{String}, cv_gradlist::Matrix{Num}, cc_gradlist::Matrix{Num})
    # Check that the exponent is supported
    ~((typeof(yL) <: Int) || (typeof(yL) <: AbstractFloat)) && error("Symbolic exponents not currently supported.")
    ~(yL == 2) && error("Exponents besides 2 not currently supported")
    
    # Identify which variables are being used
    z = findfirst(x -> x==string(zL)[1:end-3], varlist)
    x = findfirst(x -> x==string(xL)[1:end-3], varlist)
    y = findfirst(x -> x==string(yL)[1:end-3], varlist)

    # Separate variable check, in case x or y are constants
    if isnothing(x)
        x = findfirst(x -> x==string(xL), varlist)
    end
    if isnothing(y)
        y = findfirst(x -> x==string(yL), varlist)
    end

    # Helper variables
    eps_min = @. IfElse.ifelse(xU < 0.0, xU, IfElse.ifelse(xL > 0.0, xL, 0.0))
    eps_max = @. IfElse.ifelse(xU < 0.0, xL, IfElse.ifelse(xL > 0.0, xU, IfElse.ifelse(abs(xL) > abs(xU), xL, xU)))

    # Modify the gradlists accordingly
    @. cv_gradlist[:,z] = mid_grad(xcc, xcv, eps_min, cc_gradlist[:,x]*2.0*xcc, cv_gradlist[:,x]*2.0*xcv, 0.0)
    @. cc_gradlist[:,z] = mid_grad(xcc, xcv, eps_max, cc_gradlist[:,x]*IfElse.ifelse(xU > xL, xL+xU, 0.0), cv_gradlist[:,x]*IfElse.ifelse(xU > xL, xL+xU, 0.0), 0.0)
    return
end

# This one's annoying right now because of the abs(x-y), so I'm going to add min/max for grad later.
# function grad_transform!(::McCormickIntervalTransform, ::typeof(max), zL, zU, zcv, zcc, xL, xU, 
#                             xcv, xcc, yL, yU, ycv, ycc, varlist::Vector{String}, cv_gradlist::Matrix{Num}, cc_gradlist::Matrix{Num})
#     # Identify which variables are being used
#     z = findfirst(x -> x==string(zL)[1:end-3], varlist)
#     x = findfirst(x -> x==string(xL)[1:end-3], varlist)
#     y = findfirst(x -> x==string(yL)[1:end-3], varlist)

#     cc_gradlist[:,z] = IfElse.ifelse(xU <= yL, cc_gradlist[:,y], #NOTE: Regular McCormick checks the `cnst` flag, which SCMC doesn't have.
#                             IfElse.ifelse(xL >= yU, cc_gradlist[:,x], #NOTE: Regular McCormick checks the `cnst` flag, which SCMC doesn't have.
#                                 0.5*cc_gradlist[:,x]+cc_gradlist[:,y]+abs(x-y)

#                             )
#     eps_min = IfElse.ifelse(xL >= 0.0, xL, IfElse.ifelse(xU <= 0.0, xU, 0.0))
#     eps_max = IfElse.ifelse(abs(xU) >= abs(xL), xU, xL)

#     xcc, xcv, eps_max
#     midcc = IfElse.ifelse(xcc < xcv, IfElse.ifelse(xcv < c, b, IfElse.ifelse(eps_max < xcc, a, c)),
#                 IfElse.ifelse(eps_max < xcv, b, IfElse.ifelse(xcc < eps_max, a, c)))
#     whatever result is, times dcc


#     abs(x) = IfElse.ifelse(xcc < xcv, IfElse.ifelse(xcv < eps_min, IfElse.ifelse(xcv > 0.0, cv_grad[:,x], -cv_grad[:,x]), IfElse.ifelse(eps_min < xcc, IfElse.ifelse(xcc > 0.0, cc_grad[:,x], -cc_grad[:,x]), 0.0)),
#                 IfElse.ifelse(eps_min < xcv, IfElse.ifelse(xcv > 0.0, cv_grad[:,x], -cv_grad[:,x]), IfElse.ifelse(xcc < eps_min, IfElse.ifelse(xcc > 0.0, cc_grad[:,x], -cc_grad[:,x]), 0.0)))
# end



# function cut(xL::Float64, xU::Float64, cv::Float64, cc::Float64,
#              cv_grad::SVector{N,Float64}, cc_grad::SVector{N,Float64}) where N
#     if cc > xU
#         cco = xU
#         cc_grado = zero(SVector{N,Float64})
#     else
#         cco = cc
#         cc_grado = cc_grad
#     end
#     if cv < xL
#         cvo = xL
#         cv_grado = zero(SVector{N,Float64})
#     else
#         cvo = cv
#         cv_grado = cv_grad
#     end
#     return cvo, cco, cv_grado, cc_grado
# end



