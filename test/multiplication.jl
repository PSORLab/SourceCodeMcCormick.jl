
function mult_cv_case(xcv, xcc, xL, xU, ycv, ycc, yL, yU)
    if xL >= 0.0
        if yL >= 0.0
            if yU*xcv + xU*ycv - xU*yU > yL*xcv + xL*ycv - xL*yL
                return 1
            else
                return 2
            end
        elseif yU <= 0.0
            if (-yU)*xcc + xU*(-ycv) - xU*(-yU) > (-yL)*xcc + xL*(-ycv) - xL*(-yL)
                return 3
            else
                return 4
            end
        else
            if yU*xcv + xU*ycv - xU*yU > yL*xcc + xL*ycv - xL*yL
                return 5
            else
                return 6
            end
        end
    elseif xU <= 0.0
        if yL >= 0.0
            if yL*(-xcv) + (-xL)*ycc - (-xL)*yL > yU*(-xcv) + (-xU)*ycc - (-xU)*yU
                return 7
            else
                return 8
            end
        elseif yU <= 0.0
            if yL*xcc + xL*ycc - xL*yL > yU*xcc + xU*ycc - xU*yU
                return 9
            else
                return 10
            end
        else
            if yL*(-xcc) + (-xL)*ycc - (-xL)*yL > yU*(-xcv) + (-xU)*ycc - (-xU)*yU
                return 11
            else
                return 12
            end
        end
    else
        if yL >= 0.0
            if xU*ycv + yU*xcv - yU*xU > xL*ycc + yL*xcv - yL*xL
                return 13
            else
                return 14
            end
        elseif yU <= 0.0
            if xL*(-ycc) + (-yL)*xcc - (-yL)*xL > xU*(-ycv) + (-yU)*xcc - (-yU)*xU
                return 15
            else
                return 16
            end
        else
            if yU*xcv + xU*ycv - xU*yU > yL*xcc + xL*ycc - xL*yL
                return 17
            else
                return 18
            end
        end
    end
end
function mult_cc_case(xcv, xcc, xL, xU, ycv, ycc, yL, yU)
    if xL >= 0.0
        if yL >= 0.0
            if yL*xcc + xU*ycc - xU*yL > yU*xcc + xL*ycc - xL*yU
                return 1
            else
                return 2
            end
        elseif yU <= 0.0
            if (-yL)*xcv + xU*(-ycc) - xU*(-yL) > (-yU)*xcv + xL*(-ycc) - xL*(-yU)
                return 3
            else
                return 4
            end
        else
            if yL*xcv + xU*ycc - xU*yL > yU*xcc + xL*ycc - xL*yU
                return 5
            else
                return 6
            end
        end
    elseif xU <= 0.0
        if yL >= 0.0
            if yU*(-xcc) + (-xL)*ycv - (-xL)*yU > yL*(-xcc) + (-xU)*ycv - (-xU)*yL
                return 7
            else
                return 8
            end
        elseif yU <= 0.0
            if yU*xcv + xL*ycv - xL*yU > yL*xcv + xU*ycv - xU*yL
                return 9
            else
                return 10
            end
        else
            if yU*(-xcc) + (-xL)*ycv - (-xL)*yU > yL*(-xcv) + (-xU)*ycv - (-xU)*yL
                return 11
            else
                return 12
            end
        end
    else
        if yL >= 0.0
            if xL*ycv + yU*xcc - yU*xL > xU*ycc + yL*xcc - yL*xU
                return 13
            else
                return 14
            end
        elseif yU <= 0.0
            if xU*(-ycc) + (-yL)*xcv - (-yL)*xU > xL*(-ycv) + (-yU)*xcv - (-yU)*xL
                return 15
            else
                return 16
            end
        else
            if yL*xcv + xU*ycc - xU*yL > yU*xcc + xL*ycv - xL*yU
                return 17
            else
                return 18
            end
        end
    end
end
mult_cv_case(A::MC, B::MC) = mult_cv_case(A.cv, A.cc, A.Intv.lo, A.Intv.hi, B.cv, B.cc, B.Intv.lo, B.Intv.hi)
mult_cc_case(A::MC, B::MC) = mult_cv_case(A.cv, A.cc, A.Intv.lo, A.Intv.hi, B.cv, B.cc, B.Intv.lo, B.Intv.hi)

# For multiplication, need to test all cases with MC*MC, then several cases with Real types
@testset "Multiplication" begin
    @variables x, y
    
    # McCormick object times a real
    pos = MC{1,NS}(1.0, 1.5, Interval(0.5, 2.0), SVector{1, Float64}(1.0), SVector{1, Float64}(3.0), false)
    mix = MC{1,NS}(-0.5, 0.5, Interval(-2.0, 2.0), SVector{1, Float64}(-0.5), SVector{1, Float64}(1.0), false)
    neg = -pos

    to_compute = 5.0*y
    posreal_mult_cv, posreal_mult_cc, posreal_mult_lo, posreal_mult_hi, posreal_order = all_evaluators(to_compute)
    
    to_compute = -5.0*y
    negreal_mult_cv, negreal_mult_cc, negreal_mult_lo, negreal_mult_hi, negreal_order = all_evaluators(to_compute)

    # pos*pos
    @test abs(eval_check(posreal_mult_cv, pos) - (5.0*pos).cv) <= 1E-15
    @test abs(eval_check(posreal_mult_cc, pos) - (5.0*pos).cc) <= 1E-15
    @test abs(eval_check(posreal_mult_lo, pos) - (5.0*pos).Intv.lo) <= 1E-15
    @test abs(eval_check(posreal_mult_hi, pos) - (5.0*pos).Intv.hi) <= 1E-15

    # pos*mix
    @test abs(eval_check(posreal_mult_cv, mix) - (5.0*mix).cv) <= 1E-15
    @test abs(eval_check(posreal_mult_cc, mix) - (5.0*mix).cc) <= 1E-15
    @test abs(eval_check(posreal_mult_lo, mix) - (5.0*mix).Intv.lo) <= 1E-15
    @test abs(eval_check(posreal_mult_hi, mix) - (5.0*mix).Intv.hi) <= 1E-15
    
    # pos*neg
    @test abs(eval_check(posreal_mult_cv, neg) - (5.0*neg).cv) <= 1E-15
    @test abs(eval_check(posreal_mult_cc, neg) - (5.0*neg).cc) <= 1E-15
    @test abs(eval_check(posreal_mult_lo, neg) - (5.0*neg).Intv.lo) <= 1E-15
    @test abs(eval_check(posreal_mult_hi, neg) - (5.0*neg).Intv.hi) <= 1E-15

    # neg*pos
    @test abs(eval_check(negreal_mult_cv, pos) - (-5.0*pos).cv) <= 1E-15
    @test abs(eval_check(negreal_mult_cc, pos) - (-5.0*pos).cc) <= 1E-15
    @test abs(eval_check(negreal_mult_lo, pos) - (-5.0*pos).Intv.lo) <= 1E-15
    @test abs(eval_check(negreal_mult_hi, pos) - (-5.0*pos).Intv.hi) <= 1E-15

    # neg*mix
    @test abs(eval_check(negreal_mult_cv, mix) - (-5.0*mix).cv) <= 1E-15
    @test abs(eval_check(negreal_mult_cc, mix) - (-5.0*mix).cc) <= 1E-15
    @test abs(eval_check(negreal_mult_lo, mix) - (-5.0*mix).Intv.lo) <= 1E-15
    @test abs(eval_check(negreal_mult_hi, mix) - (-5.0*mix).Intv.hi) <= 1E-15
    
    # neg*neg
    @test abs(eval_check(negreal_mult_cv, neg) - (-5.0*neg).cv) <= 1E-15
    @test abs(eval_check(negreal_mult_cc, neg) - (-5.0*neg).cc) <= 1E-15
    @test abs(eval_check(negreal_mult_lo, neg) - (-5.0*neg).Intv.lo) <= 1E-15
    @test abs(eval_check(negreal_mult_hi, neg) - (-5.0*neg).Intv.hi) <= 1E-15


    # McCormick object times a McCormick object. Verify that cases are satisfied before
    # checking the solution using the cv/cc case checker functions
    pos = MC{2,NS}(1.0, 1.5, Interval(0.5, 2.0), SVector{2, Float64}(1.0, 2.0), SVector{2, Float64}(3.0, 4.0), false)
    pos_hi = MC{2,NS}(1.8, 1.9, Interval(0.5, 2.0), SVector{2, Float64}(1.0, 2.0), SVector{2, Float64}(3.0, 4.0), false)
    mix = MC{2,NS}(-0.5, 0.5, Interval(-2.0, 2.0), SVector{2, Float64}(-0.5, 0.5), SVector{2, Float64}(1.0, -1.0), false)
    mix_lo = MC{2,NS}(-0.5, 0.5, Interval(-2.0, 1.0), SVector{2, Float64}(-0.5, 0.5), SVector{2, Float64}(1.0, -1.0), false)
    mix_hi = MC{2,NS}(-0.5, 0.5, Interval(-2.0, 5.0), SVector{2, Float64}(-0.5, 0.5), SVector{2, Float64}(1.0, -1.0), false)
    neg = -pos
    neg_hi = -pos_hi

    @variables x, y
    to_compute = x*y
    mult_cv, mult_cc, mult_lo, mult_hi, order = all_evaluators(to_compute)

    @test mult_cv_case(pos, pos_hi) == 1
    @test mult_cc_case(pos, pos_hi) == 1
    @test abs(eval_check(mult_cv, pos, pos_hi) - (pos*pos_hi).cv) <= 1E-15
    @test abs(eval_check(mult_cc, pos, pos_hi) - (pos*pos_hi).cc) <= 1E-15
    @test abs(eval_check(mult_lo, pos, pos_hi) - (pos*pos_hi).Intv.lo) <= 1E-15
    @test abs(eval_check(mult_hi, pos, pos_hi) - (pos*pos_hi).Intv.hi) <= 1E-15
    
    @test mult_cv_case(pos, pos) == 2
    @test mult_cc_case(pos, pos) == 2
    @test abs(eval_check(mult_cv, pos, pos) - (pos*pos).cv) <= 1E-15
    @test abs(eval_check(mult_cc, pos, pos) - (pos*pos).cc) <= 1E-15
    @test abs(eval_check(mult_lo, pos, pos) - (pos*pos).Intv.lo) <= 1E-15
    @test abs(eval_check(mult_hi, pos, pos) - (pos*pos).Intv.hi) <= 1E-15
    
    @test mult_cv_case(pos, neg_hi) == 3
    @test mult_cc_case(pos, neg_hi) == 3
    @test abs(eval_check(mult_cv, pos, neg_hi) - (pos*neg_hi).cv) <= 1E-15
    @test abs(eval_check(mult_cc, pos, neg_hi) - (pos*neg_hi).cc) <= 1E-15
    @test abs(eval_check(mult_lo, pos, neg_hi) - (pos*neg_hi).Intv.lo) <= 1E-15
    @test abs(eval_check(mult_hi, pos, neg_hi) - (pos*neg_hi).Intv.hi) <= 1E-15
    
    @test mult_cv_case(pos, neg) == 4
    @test mult_cc_case(pos, neg) == 4
    @test abs(eval_check(mult_cv, pos, neg) - (pos*neg).cv) <= 1E-15
    @test abs(eval_check(mult_cc, pos, neg) - (pos*neg).cc) <= 1E-15
    @test abs(eval_check(mult_lo, pos, neg) - (pos*neg).Intv.lo) <= 1E-15
    @test abs(eval_check(mult_hi, pos, neg) - (pos*neg).Intv.hi) <= 1E-15
    
    @test mult_cv_case(pos_hi, mix) == 5
    @test mult_cc_case(pos_hi, mix) == 5
    @test abs(eval_check(mult_cv, pos_hi, mix) - (pos_hi*mix).cv) <= 1E-15
    @test abs(eval_check(mult_cc, pos_hi, mix) - (pos_hi*mix).cc) <= 1E-15
    @test abs(eval_check(mult_lo, pos_hi, mix) - (pos_hi*mix).Intv.lo) <= 1E-15
    @test abs(eval_check(mult_hi, pos_hi, mix) - (pos_hi*mix).Intv.hi) <= 1E-15
    
    @test mult_cv_case(pos, mix) == 6
    @test mult_cc_case(pos, mix) == 6
    @test abs(eval_check(mult_cv, pos, mix) - (pos*mix).cv) <= 1E-15
    @test abs(eval_check(mult_cc, pos, mix) - (pos*mix).cc) <= 1E-15
    @test abs(eval_check(mult_lo, pos, mix) - (pos*mix).Intv.lo) <= 1E-15
    @test abs(eval_check(mult_hi, pos, mix) - (pos*mix).Intv.hi) <= 1E-15
    
    @test mult_cv_case(neg, pos_hi) == 7
    @test mult_cc_case(neg, pos_hi) == 7
    @test abs(eval_check(mult_cv, neg, pos_hi) - (neg*pos_hi).cv) <= 1E-15
    @test abs(eval_check(mult_cc, neg, pos_hi) - (neg*pos_hi).cc) <= 1E-15
    @test abs(eval_check(mult_lo, neg, pos_hi) - (neg*pos_hi).Intv.lo) <= 1E-15
    @test abs(eval_check(mult_hi, neg, pos_hi) - (neg*pos_hi).Intv.hi) <= 1E-15
    
    @test mult_cv_case(neg, pos) == 8
    @test mult_cc_case(neg, pos) == 8
    @test abs(eval_check(mult_cv, neg, pos) - (neg*pos).cv) <= 1E-15
    @test abs(eval_check(mult_cc, neg, pos) - (neg*pos).cc) <= 1E-15
    @test abs(eval_check(mult_lo, neg, pos) - (neg*pos).Intv.lo) <= 1E-15
    @test abs(eval_check(mult_hi, neg, pos) - (neg*pos).Intv.hi) <= 1E-15
    
    @test mult_cv_case(neg, neg_hi) == 9
    @test mult_cc_case(neg, neg_hi) == 9
    @test abs(eval_check(mult_cv, neg, neg_hi) - (neg*neg_hi).cv) <= 1E-15
    @test abs(eval_check(mult_cc, neg, neg_hi) - (neg*neg_hi).cc) <= 1E-15
    @test abs(eval_check(mult_lo, neg, neg_hi) - (neg*neg_hi).Intv.lo) <= 1E-15
    @test abs(eval_check(mult_hi, neg, neg_hi) - (neg*neg_hi).Intv.hi) <= 1E-15
    
    @test mult_cv_case(neg, neg) == 10
    @test mult_cc_case(neg, neg) == 10
    @test abs(eval_check(mult_cv, neg, neg) - (neg*neg).cv) <= 1E-15
    @test abs(eval_check(mult_cc, neg, neg) - (neg*neg).cc) <= 1E-15
    @test abs(eval_check(mult_lo, neg, neg) - (neg*neg).Intv.lo) <= 1E-15
    @test abs(eval_check(mult_hi, neg, neg) - (neg*neg).Intv.hi) <= 1E-15
    
    @test mult_cv_case(neg, mix) == 11
    @test mult_cc_case(neg, mix) == 11
    @test abs(eval_check(mult_cv, neg, mix) - (neg*mix).cv) <= 1E-15
    @test abs(eval_check(mult_cc, neg, mix) - (neg*mix).cc) <= 1E-15
    @test abs(eval_check(mult_lo, neg, mix) - (neg*mix).Intv.lo) <= 1E-15
    @test abs(eval_check(mult_hi, neg, mix) - (neg*mix).Intv.hi) <= 1E-15
    
    @test mult_cv_case(neg_hi, mix) == 12
    @test mult_cc_case(neg_hi, mix) == 12
    @test abs(eval_check(mult_cv, neg_hi, mix) - (neg_hi*mix).cv) <= 1E-15
    @test abs(eval_check(mult_cc, neg_hi, mix) - (neg_hi*mix).cc) <= 1E-15
    @test abs(eval_check(mult_lo, neg_hi, mix) - (neg_hi*mix).Intv.lo) <= 1E-15
    @test abs(eval_check(mult_hi, neg_hi, mix) - (neg_hi*mix).Intv.hi) <= 1E-15
    
    @test mult_cv_case(mix, pos_hi) == 13
    @test mult_cc_case(mix, pos_hi) == 13
    @test abs(eval_check(mult_cv, mix, pos_hi) - (mix*pos_hi).cv) <= 1E-15
    @test abs(eval_check(mult_cc, mix, pos_hi) - (mix*pos_hi).cc) <= 1E-15
    @test abs(eval_check(mult_lo, mix, pos_hi) - (mix*pos_hi).Intv.lo) <= 1E-15
    @test abs(eval_check(mult_hi, mix, pos_hi) - (mix*pos_hi).Intv.hi) <= 1E-15
    
    @test mult_cv_case(mix, pos) == 14
    @test mult_cc_case(mix, pos) == 14
    @test abs(eval_check(mult_cv, mix, pos) - (mix*pos).cv) <= 1E-15
    @test abs(eval_check(mult_cc, mix, pos) - (mix*pos).cc) <= 1E-15
    @test abs(eval_check(mult_lo, mix, pos) - (mix*pos).Intv.lo) <= 1E-15
    @test abs(eval_check(mult_hi, mix, pos) - (mix*pos).Intv.hi) <= 1E-15
    
    @test mult_cv_case(mix, neg) == 15
    @test mult_cc_case(mix, neg) == 15
    @test abs(eval_check(mult_cv, mix, neg) - (mix*neg).cv) <= 1E-15
    @test abs(eval_check(mult_cc, mix, neg) - (mix*neg).cc) <= 1E-15
    @test abs(eval_check(mult_lo, mix, neg) - (mix*neg).Intv.lo) <= 1E-15
    @test abs(eval_check(mult_hi, mix, neg) - (mix*neg).Intv.hi) <= 1E-15
    
    @test mult_cv_case(mix, neg_hi) == 16
    @test mult_cc_case(mix, neg_hi) == 16
    @test abs(eval_check(mult_cv, mix, neg_hi) - (mix*neg_hi).cv) <= 1E-15
    @test abs(eval_check(mult_cc, mix, neg_hi) - (mix*neg_hi).cc) <= 1E-15
    @test abs(eval_check(mult_lo, mix, neg_hi) - (mix*neg_hi).Intv.lo) <= 1E-15
    @test abs(eval_check(mult_hi, mix, neg_hi) - (mix*neg_hi).Intv.hi) <= 1E-15
    
    @test mult_cv_case(mix, mix_lo) == 17
    @test mult_cc_case(mix, mix_lo) == 17
    @test abs(eval_check(mult_cv, mix, mix_lo) - (mix*mix_lo).cv) <= 1E-15
    @test abs(eval_check(mult_cc, mix, mix_lo) - (mix*mix_lo).cc) <= 1E-15
    @test abs(eval_check(mult_lo, mix, mix_lo) - (mix*mix_lo).Intv.lo) <= 1E-15
    @test abs(eval_check(mult_hi, mix, mix_lo) - (mix*mix_lo).Intv.hi) <= 1E-15
    
    @test mult_cv_case(mix, mix_hi) == 18
    @test mult_cc_case(mix, mix_hi) == 18
    @test abs(eval_check(mult_cv, mix, mix_hi) - (mix*mix_hi).cv) <= 1E-15
    @test abs(eval_check(mult_cc, mix, mix_hi) - (mix*mix_hi).cc) <= 1E-15
    @test abs(eval_check(mult_lo, mix, mix_hi) - (mix*mix_hi).Intv.lo) <= 1E-15
    @test abs(eval_check(mult_hi, mix, mix_hi) - (mix*mix_hi).Intv.hi) <= 1E-15
end
