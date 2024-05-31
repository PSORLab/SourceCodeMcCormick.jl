mid_expr(x, y, z) = IfElse.ifelse(x >= y, IfElse.ifelse(y >= z, y, IfElse.ifelse(y == x, y, IfElse.ifelse(z >= x, x, z))),
        IfElse.ifelse(z >= y, y, IfElse.ifelse(x >= z, x, z)))

function div_cv_case(xcv, xcc, xL, xU, ycv, ycc, yL, yU)
    yL_inv = inv(yU)
    yU_inv = inv(yL)
    ycv_inv = ifelse(yL > 0.0, ifelse(yU <= ycv, 1.0 ./ ycv, 
            ifelse(yU >= ycc, 1.0 ./ ycc, 1.0 ./ yU)),
        ifelse(yU < 0.0, ifelse(yL == yU, mid_expr(ycc, ycv, yL).^(-1), 
            ((yL.^(-1))*(yU - mid_expr(ycc, ycv, yL)) + (yU.^(-1))*(mid_expr(ycc, ycv, yL) - yL))./(yU - yL)),
                NaN))
    ycc_inv = ifelse(yL > 0.0, ifelse(yL <= ycv, (yU + yL - ycv)./(yL*yU), 
                ifelse(yL >= ycc, (yU + yL - ycc)./(yL*yU), 1.0 ./ yL)),
            ifelse(yU < 0.0, mid_expr(ycc, ycv, yU).^(-1),
                NaN))
    if xL >= 0.0
        if yL >= 0.0
            if yU_inv*xcv + xU*ycv_inv - xU*yU_inv > yL_inv*xcv + xL*ycv_inv - xL*yL_inv
                return 1
            else
                return 2
            end
        elseif yU <= 0.0
            if (-yU_inv)*xcc + xU*(-ycv_inv) - xU*(-yU_inv) > (-yL_inv)*xcc + xL*(-ycv_inv) - xL*(-yL_inv)
                return 3
            else
                return 4
            end
        else
            return 5
        end
    elseif xU <= 0.0
        if yL >= 0.0
            if yL_inv*(-xcv) + (-xL)*ycc_inv - (-xL)*yL_inv > yU_inv*(-xcv) + (-xU)*ycc_inv - (-xU)*yU_inv
                return 6
            else
                return 7
            end
        elseif yU <= 0.0
            if yL_inv*xcc + xL*ycc_inv - xL*yL_inv > yU_inv*xcc + xU*ycc_inv - xU*yU_inv
                return 8
            else
                return 9
            end
        else
            return 10
        end
    else
        if yL >= 0.0
            if xU*ycv_inv + yU_inv*xcv - yU_inv*xU > xL*ycc_inv + yL_inv*xcv - yL_inv*xL
                return 11
            else
                return 12
            end
        elseif yU <= 0.0
            if xL*(-ycc_inv) + (-yL_inv)*xcc - (-yL_inv)*xL > xU*(-ycv_inv) + (-yU_inv)*xcc - (-yU_inv)*xU
                return 13
            else
                return 14
            end
        else
            return 15
        end
    end
end
function div_cc_case(xcv, xcc, xL, xU, ycv, ycc, yL, yU)
    yL_inv = inv(yU)
    yU_inv = inv(yL)
    ycv_inv = ifelse(yL > 0.0, ifelse(yU <= ycv, 1.0 ./ ycv, 
            ifelse(yU >= ycc, 1.0 ./ ycc, 1.0 ./ yU)),
        ifelse(yU < 0.0, ifelse(yL == yU, mid_expr(ycc, ycv, yL).^(-1), 
            ((yL.^(-1))*(yU - mid_expr(ycc, ycv, yL)) + (yU.^(-1))*(mid_expr(ycc, ycv, yL) - yL))./(yU - yL)),
                NaN))
    ycc_inv = ifelse(yL > 0.0, ifelse(yL <= ycv, (yU + yL - ycv)./(yL*yU), 
                ifelse(yL >= ycc, (yU + yL - ycc)./(yL*yU), 1.0 ./ yL)),
            ifelse(yU < 0.0, mid_expr(ycc, ycv, yU).^(-1),
                NaN))
    if xL >= 0.0
        if yL >= 0.0
            if yL_inv*xcc + xU*ycc_inv - xU*yL_inv > yU_inv*xcc + xL*ycc_inv - xL*yU_inv
                return 1
            else
                return 2
            end
        elseif yU <= 0.0
            if (-yL_inv)*xcv + xU*(-ycc_inv) - xU*(-yL_inv) > (-yU_inv)*xcv + xL*(-ycc_inv) - xL*(-yU_inv)
                return 3
            else
                return 4
            end
        else
            return 5
        end
    elseif xU <= 0.0
        if yL >= 0.0
            if yU_inv*(-xcc) + (-xL)*ycv_inv - (-xL)*yU_inv > yL_inv*(-xcc) + (-xU)*ycv_inv - (-xU)*yL_inv
                return 6
            else
                return 7
            end
        elseif yU <= 0.0
            if yU_inv*xcv + xL*ycv_inv - xL*yU_inv > yL_inv*xcv + xU*ycv_inv - xU*yL_inv
                return 8
            else
                return 9
            end
        else
            return 10
        end
    else
        if yL >= 0.0
            if xL*ycv_inv + yU_inv*xcc - yU_inv*xL > xU*ycc_inv + yL_inv*xcc - yL_inv*xU
                return 11
            else
                return 12
            end
        elseif yU <= 0.0
            if xU*(-ycc_inv) + (-yL_inv)*xcv - (-yL_inv)*xU > xL*(-ycv_inv) + (-yU_inv)*xcv - (-yU_inv)*xL
                return 13
            else
                return 14
            end
        else
            return 15
        end
    end
end
div_cv_case(A::MC, B::MC) = div_cv_case(A.cv, A.cc, A.Intv.lo, A.Intv.hi, B.cv, B.cc, B.Intv.lo, B.Intv.hi)
div_cc_case(A::MC, B::MC) = div_cv_case(A.cv, A.cc, A.Intv.lo, A.Intv.hi, B.cv, B.cc, B.Intv.lo, B.Intv.hi)

# For division, need to test all cases with MC*MC, then several cases with Real types
@testset "Division" begin
    @variables x, y
    
    # Real divided by a McCormick object
    pos = MC{1,NS}(1.0, 1.5, Interval(0.5, 2.0), SVector{1, Float64}(1.0), SVector{1, Float64}(3.0), false)
    mix = MC{1,NS}(-0.5, 0.5, Interval(-2.0, 2.0), SVector{1, Float64}(-0.5), SVector{1, Float64}(1.0), false)
    neg = -pos

    to_compute = 5.0/y
    posreal_div_cv, posreal_div_cc, posreal_div_lo, posreal_div_hi, posreal_order = all_evaluators(to_compute)
    
    to_compute = -5.0/y
    negreal_div_cv, negreal_div_cc, negreal_div_lo, negreal_div_hi, negreal_order = all_evaluators(to_compute)

    # pos/pos
    @test abs(eval_check(posreal_div_cv, pos) - (5.0/pos).cv) <= 1E-15
    @test abs(eval_check(posreal_div_cc, pos) - (5.0/pos).cc) <= 1E-15
    @test abs(eval_check(posreal_div_lo, pos) - (5.0/pos).Intv.lo) <= 1E-15
    @test abs(eval_check(posreal_div_hi, pos) - (5.0/pos).Intv.hi) <= 1E-15

    # pos/mix
    @test isnan(eval_check(posreal_div_cv, mix))
    @test isnan((5.0/mix).cv)
    @test isnan(eval_check(posreal_div_cc, mix))
    @test isnan((5.0/mix).cc)
    @test isnan(eval_check(posreal_div_lo, mix))
    @test isnan((5.0/mix).Intv.lo)
    @test isnan(eval_check(posreal_div_hi, mix))
    @test isnan((5.0/mix).Intv.hi)
    
    # pos/neg
    @test abs(eval_check(posreal_div_cv, neg) - (5.0/neg).cv) <= 1E-15
    @test abs(eval_check(posreal_div_cc, neg) - (5.0/neg).cc) <= 1E-15
    @test abs(eval_check(posreal_div_lo, neg) - (5.0/neg).Intv.lo) <= 1E-15
    @test abs(eval_check(posreal_div_hi, neg) - (5.0/neg).Intv.hi) <= 1E-15

    # neg/pos
    @test abs(eval_check(negreal_div_cv, pos) - (-5.0/pos).cv) <= 1E-15
    @test abs(eval_check(negreal_div_cc, pos) - (-5.0/pos).cc) <= 1E-15
    @test abs(eval_check(negreal_div_lo, pos) - (-5.0/pos).Intv.lo) <= 1E-15
    @test abs(eval_check(negreal_div_hi, pos) - (-5.0/pos).Intv.hi) <= 1E-15

    # neg/mix
    @test isnan(eval_check(negreal_div_cv, mix))
    @test isnan((-5.0/mix).cv)
    @test isnan(eval_check(negreal_div_cc, mix))
    @test isnan((-5.0/mix).cc)
    @test isnan(eval_check(negreal_div_lo, mix))
    @test isnan((-5.0/mix).Intv.lo)
    @test isnan(eval_check(negreal_div_hi, mix))
    @test isnan((-5.0/mix).Intv.hi)
    
    # neg/neg
    @test abs(eval_check(negreal_div_cv, neg) - (-5.0/neg).cv) <= 1E-15
    @test abs(eval_check(negreal_div_cc, neg) - (-5.0/neg).cc) <= 1E-15
    @test abs(eval_check(negreal_div_lo, neg) - (-5.0/neg).Intv.lo) <= 1E-15
    @test abs(eval_check(negreal_div_hi, neg) - (-5.0/neg).Intv.hi) <= 1E-15

    # Note: McCormick object divided by a real automatically converts from (MC/real) to (MC*(real^-1))
    # through Symbolics.jl, so this is simply multiplication.


    # McCormick object divided by a McCormick object. Verify that cases are satisfied before
    # checking the solution using the cv/cc case checker functions
    pos = MC{2,NS}(1.0, 1.5, Interval(0.5, 2.0), SVector{2, Float64}(1.0, 2.0), SVector{2, Float64}(3.0, 4.0), false)
    pos_lo = MC{2,NS}(0.5, 0.5, Interval(0.5, 2.0), SVector{2, Float64}(1.0, 2.0), SVector{2, Float64}(3.0, 4.0), false)
    pos_hi = MC{2,NS}(1.8, 1.9, Interval(0.5, 2.0), SVector{2, Float64}(1.0, 2.0), SVector{2, Float64}(3.0, 4.0), false)
    mix = MC{2,NS}(-0.5, 0.5, Interval(-2.0, 2.0), SVector{2, Float64}(-0.5, 0.5), SVector{2, Float64}(1.0, -1.0), false)
    neg = -pos
    neg_lo = -pos_lo
    neg_hi = -pos_hi

    @variables x, y
    to_compute = x/y
    div_cv, div_cc, div_lo, div_hi, order = all_evaluators(to_compute)

    @test div_cv_case(pos, pos_lo) == 1
    @test div_cc_case(pos, pos_lo) == 1
    @test abs(eval_check(div_cv, pos, pos_lo) - (pos/pos_lo).cv) <= 1E-15
    @test abs(eval_check(div_cc, pos, pos_lo) - (pos/pos_lo).cc) <= 1E-15
    @test abs(eval_check(div_lo, pos, pos_lo) - (pos/pos_lo).Intv.lo) <= 1E-15
    @test abs(eval_check(div_hi, pos, pos_lo) - (pos/pos_lo).Intv.hi) <= 1E-15
    
    @test div_cv_case(pos, pos_hi) == 2
    @test div_cc_case(pos, pos_hi) == 2
    @test abs(eval_check(div_cv, pos, pos_hi) - (pos/pos_hi).cv) <= 1E-15
    @test abs(eval_check(div_cc, pos, pos_hi) - (pos/pos_hi).cc) <= 1E-15
    @test abs(eval_check(div_lo, pos, pos_hi) - (pos/pos_hi).Intv.lo) <= 1E-15
    @test abs(eval_check(div_hi, pos, pos_hi) - (pos/pos_hi).Intv.hi) <= 1E-15
    
    @test div_cv_case(pos, neg_lo) == 3
    @test div_cc_case(pos, neg_lo) == 3
    @test abs(eval_check(div_cv, pos, neg_lo) - (pos/neg_lo).cv) <= 1E-15
    @test abs(eval_check(div_cc, pos, neg_lo) - (pos/neg_lo).cc) <= 1E-15
    @test abs(eval_check(div_lo, pos, neg_lo) - (pos/neg_lo).Intv.lo) <= 1E-15
    @test abs(eval_check(div_hi, pos, neg_lo) - (pos/neg_lo).Intv.hi) <= 1E-15
    
    @test div_cv_case(pos, neg) == 4
    @test div_cc_case(pos, neg) == 4
    @test abs(eval_check(div_cv, pos, neg) - (pos/neg).cv) <= 1E-15
    @test abs(eval_check(div_cc, pos, neg) - (pos/neg).cc) <= 1E-15
    @test abs(eval_check(div_lo, pos, neg) - (pos/neg).Intv.lo) <= 1E-15
    @test abs(eval_check(div_hi, pos, neg) - (pos/neg).Intv.hi) <= 1E-15
    
    @test div_cv_case(pos_hi, mix) == 5
    @test div_cc_case(pos_hi, mix) == 5
    @test isnan(eval_check(div_cv, pos_hi, mix))
    @test isnan((pos_hi/mix).cv)
    @test isnan(eval_check(div_cc, pos_hi, mix))
    @test isnan((pos_hi/mix).cc)
    @test isnan(eval_check(div_lo, pos_hi, mix))
    @test isnan((pos_hi/mix).Intv.lo)
    @test isnan(eval_check(div_hi, pos_hi, mix))
    @test isnan((pos_hi/mix).Intv.hi)

    @test div_cv_case(neg, pos_lo) == 6
    @test div_cc_case(neg, pos_lo) == 6
    @test abs(eval_check(div_cv, neg, pos_lo) - (neg/pos_lo).cv) <= 1E-15
    @test abs(eval_check(div_cc, neg, pos_lo) - (neg/pos_lo).cc) <= 1E-15
    @test abs(eval_check(div_lo, neg, pos_lo) - (neg/pos_lo).Intv.lo) <= 1E-15
    @test abs(eval_check(div_hi, neg, pos_lo) - (neg/pos_lo).Intv.hi) <= 1E-15
    
    @test div_cv_case(neg, pos_hi) == 7
    @test div_cc_case(neg, pos_hi) == 7
    @test abs(eval_check(div_cv, neg, pos_hi) - (neg/pos_hi).cv) <= 1E-15
    @test abs(eval_check(div_cc, neg, pos_hi) - (neg/pos_hi).cc) <= 1E-15
    @test abs(eval_check(div_lo, neg, pos_hi) - (neg/pos_hi).Intv.lo) <= 1E-15
    @test abs(eval_check(div_hi, neg, pos_hi) - (neg/pos_hi).Intv.hi) <= 1E-15
    
    @test div_cv_case(neg, neg_lo) == 8
    @test div_cc_case(neg, neg_lo) == 8
    @test abs(eval_check(div_cv, neg, neg_lo) - (neg/neg_lo).cv) <= 1E-15
    @test abs(eval_check(div_cc, neg, neg_lo) - (neg/neg_lo).cc) <= 1E-15
    @test abs(eval_check(div_lo, neg, neg_lo) - (neg/neg_lo).Intv.lo) <= 1E-15
    @test abs(eval_check(div_hi, neg, neg_lo) - (neg/neg_lo).Intv.hi) <= 1E-15
    
    @test div_cv_case(neg, neg_hi) == 9
    @test div_cc_case(neg, neg_hi) == 9
    @test abs(eval_check(div_cv, neg, neg_hi) - (neg/neg_hi).cv) <= 1E-15
    @test abs(eval_check(div_cc, neg, neg_hi) - (neg/neg_hi).cc) <= 1E-15
    @test abs(eval_check(div_lo, neg, neg_hi) - (neg/neg_hi).Intv.lo) <= 1E-15
    @test abs(eval_check(div_hi, neg, neg_hi) - (neg/neg_hi).Intv.hi) <= 1E-15
    
    @test div_cv_case(neg, mix) == 10
    @test div_cc_case(neg, mix) == 10
    @test isnan(eval_check(div_cv, neg, mix))
    @test isnan((neg/mix).cv)
    @test isnan(eval_check(div_cc, neg, mix))
    @test isnan((neg/mix).cc)
    @test isnan(eval_check(div_lo, neg, mix))
    @test isnan((neg/mix).Intv.lo)
    @test isnan(eval_check(div_hi, neg, mix))
    @test isnan((neg/mix).Intv.hi)

    @test div_cv_case(mix, pos_lo) == 11
    @test div_cc_case(mix, pos_lo) == 11
    @test abs(eval_check(div_cv, mix, pos_lo) - (mix/pos_lo).cv) <= 1E-15
    @test abs(eval_check(div_cc, mix, pos_lo) - (mix/pos_lo).cc) <= 1E-15
    @test abs(eval_check(div_lo, mix, pos_lo) - (mix/pos_lo).Intv.lo) <= 1E-15
    @test abs(eval_check(div_hi, mix, pos_lo) - (mix/pos_lo).Intv.hi) <= 1E-15
    
    @test div_cv_case(mix, pos) == 12
    @test div_cc_case(mix, pos) == 12
    @test abs(eval_check(div_cv, mix, pos) - (mix/pos).cv) <= 1E-15
    @test abs(eval_check(div_cc, mix, pos) - (mix/pos).cc) <= 1E-15
    @test abs(eval_check(div_lo, mix, pos) - (mix/pos).Intv.lo) <= 1E-15
    @test abs(eval_check(div_hi, mix, pos) - (mix/pos).Intv.hi) <= 1E-15

    @test div_cv_case(mix, neg) == 13
    @test div_cc_case(mix, neg) == 13
    @test abs(eval_check(div_cv, mix, neg) - (mix/neg).cv) <= 1E-15
    @test abs(eval_check(div_cc, mix, neg) - (mix/neg).cc) <= 1E-15
    @test abs(eval_check(div_lo, mix, neg) - (mix/neg).Intv.lo) <= 1E-15
    @test abs(eval_check(div_hi, mix, neg) - (mix/neg).Intv.hi) <= 1E-15

    @test div_cv_case(mix, neg_lo) == 14
    @test div_cc_case(mix, neg_lo) == 14
    @test abs(eval_check(div_cv, mix, neg_lo) - (mix/neg_lo).cv) <= 1E-15
    @test abs(eval_check(div_cc, mix, neg_lo) - (mix/neg_lo).cc) <= 1E-15
    @test abs(eval_check(div_lo, mix, neg_lo) - (mix/neg_lo).Intv.lo) <= 1E-15
    @test abs(eval_check(div_hi, mix, neg_lo) - (mix/neg_lo).Intv.hi) <= 1E-15
    
    @test div_cv_case(mix, mix) == 15
    @test div_cc_case(mix, mix) == 15
    @test isnan(eval_check(div_cv, mix, mix))
    @test isnan((mix/mix).cv)
    @test isnan(eval_check(div_cc, mix, mix))
    @test isnan((mix/mix).cc)
    @test isnan(eval_check(div_lo, mix, mix))
    @test isnan((mix/mix).Intv.lo)
    @test isnan(eval_check(div_hi, mix, mix))
    @test isnan((mix/mix).Intv.hi)
end