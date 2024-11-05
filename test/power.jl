
@testset "Power" begin
    # NOTE: Currently only includes ^2
    @variables x

    to_compute = x^2
    pow2_cv, pow2_cc, pow2_lo, pow2_hi, pow2_order = all_evaluators(to_compute)
    pow2_cvgrad, pow2_ccgrad, pow2_order_grad = all_subgradients(to_compute, expand=true)

    # All cases for ^2 are very similar; check positive/negative/mixed, as well as some unique cases where values are the same
    pos = MC{1,NS}(1.0, 1.5, Interval(0.5, 2.0), SVector{1, Float64}(1.0), SVector{1, Float64}(3.0), false)
    mix = MC{1,NS}(-0.5, 0.5, Interval(-2.0, 3.0), SVector{1, Float64}(-0.5), SVector{1, Float64}(1.0), false)
    neg = -pos
    pos_same1 = MC{1,NS}(1.0, 1.0, Interval(0.5, 2.0), SVector{1, Float64}(1.0), SVector{1, Float64}(3.0), false)
    pos_same2 = MC{1,NS}(1.0, 1.0, Interval(1.0, 2.0), SVector{1, Float64}(1.0), SVector{1, Float64}(3.0), false)
    pos_same3 = MC{1,NS}(1.0, 1.0, Interval(1.0, 1.0), SVector{1, Float64}(1.0), SVector{1, Float64}(1.0), false)

    # ADD DEGENERATE POSITIVE/NEGATIVE/ZERO??

    @test abs(eval_check(pow2_cv, pos) - (pos^2).cv) <= 1E-15
    @test abs(eval_check(pow2_cc, pos) - (pos^2).cc) <= 1E-15
    @test abs(eval_check(pow2_lo, pos) - (pos^2).Intv.lo) <= 1E-15
    @test abs(eval_check(pow2_hi, pos) - (pos^2).Intv.hi) <= 1E-15
    @test abs(eval_check_grad(pow2_cvgrad[1], pos) - (pos^2).cv_grad[1]) <= 1E-15
    @test abs(eval_check_grad(pow2_ccgrad[1], pos) - (pos^2).cc_grad[1]) <= 1E-15
    
    @test abs(eval_check(pow2_cv, mix) - (mix^2).cv) <= 1E-15
    @test abs(eval_check(pow2_cc, mix) - (mix^2).cc) <= 1E-15
    @test abs(eval_check(pow2_lo, mix) - (mix^2).Intv.lo) <= 1E-15
    @test abs(eval_check(pow2_hi, mix) - (mix^2).Intv.hi) <= 1E-15
    @test abs(eval_check_grad(pow2_cvgrad[1], mix) - (mix^2).cv_grad[1]) <= 1E-15
    @test abs(eval_check_grad(pow2_ccgrad[1], mix) - (mix^2).cc_grad[1]) <= 1E-15
    
    @test abs(eval_check(pow2_cv, neg) - (neg^2).cv) <= 1E-15
    @test abs(eval_check(pow2_cc, neg) - (neg^2).cc) <= 1E-15
    @test abs(eval_check(pow2_lo, neg) - (neg^2).Intv.lo) <= 1E-15
    @test abs(eval_check(pow2_hi, neg) - (neg^2).Intv.hi) <= 1E-15
    @test abs(eval_check_grad(pow2_cvgrad[1], neg) - (neg^2).cv_grad[1]) <= 1E-15
    @test abs(eval_check_grad(pow2_ccgrad[1], neg) - (neg^2).cc_grad[1]) <= 1E-15

    @test abs(eval_check(pow2_cv, pos_same1) - (pos_same1^2).cv) <= 1E-15
    @test abs(eval_check(pow2_cc, pos_same1) - (pos_same1^2).cc) <= 1E-15
    @test abs(eval_check(pow2_lo, pos_same1) - (pos_same1^2).Intv.lo) <= 1E-15
    @test abs(eval_check(pow2_hi, pos_same1) - (pos_same1^2).Intv.hi) <= 1E-15
    @test abs(eval_check_grad(pow2_cvgrad[1], pos_same1) - (pos_same1^2).cv_grad[1]) <= 1E-15
    @test abs(eval_check_grad(pow2_ccgrad[1], pos_same1) - (pos_same1^2).cc_grad[1]) <= 1E-15

    @test abs(eval_check(pow2_cv, pos_same2) - (pos_same2^2).cv) <= 1E-15
    @test abs(eval_check(pow2_cc, pos_same2) - (pos_same2^2).cc) <= 1E-15
    @test abs(eval_check(pow2_lo, pos_same2) - (pos_same2^2).Intv.lo) <= 1E-15
    @test abs(eval_check(pow2_hi, pos_same2) - (pos_same2^2).Intv.hi) <= 1E-15
    @test abs(eval_check_grad(pow2_cvgrad[1], pos_same2) - (pos_same2^2).cv_grad[1]) <= 1E-15
    @test abs(eval_check_grad(pow2_ccgrad[1], pos_same2) - (pos_same2^2).cc_grad[1]) <= 1E-15

    @test abs(eval_check(pow2_cv, pos_same3) - (pos_same3^2).cv) <= 1E-15
    @test abs(eval_check(pow2_cc, pos_same3) - (pos_same3^2).cc) <= 1E-15
    @test abs(eval_check(pow2_lo, pos_same3) - (pos_same3^2).Intv.lo) <= 1E-15
    @test abs(eval_check(pow2_hi, pos_same3) - (pos_same3^2).Intv.hi) <= 1E-15
    @test abs(eval_check_grad(pow2_cvgrad[1], pos_same3) - (pos_same3^2).cv_grad[1]) <= 1E-15
    @test abs(eval_check_grad(pow2_ccgrad[1], pos_same3) - (pos_same3^2).cc_grad[1]) <= 1E-15
end
