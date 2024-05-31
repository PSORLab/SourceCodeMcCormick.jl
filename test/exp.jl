
@testset "Exponential" begin
    @variables x

    to_compute = exp(x)
    exp_cv, exp_cc, exp_lo, exp_hi, exp_order = all_evaluators(to_compute)

    # Check positive/negative/mixed cases, as well as some unique cases where values are the same
    pos = MC{1,NS}(1.0, 1.5, Interval(0.5, 2.0), SVector{1, Float64}(1.0), SVector{1, Float64}(3.0), false)
    mix = MC{1,NS}(-0.5, 0.5, Interval(-2.0, 3.0), SVector{1, Float64}(-0.5), SVector{1, Float64}(1.0), false)
    neg = -pos
    pos_same1 = MC{1,NS}(1.0, 1.0, Interval(0.5, 2.0), SVector{1, Float64}(1.0), SVector{1, Float64}(3.0), false)
    pos_same2 = MC{1,NS}(1.0, 1.0, Interval(1.0, 2.0), SVector{1, Float64}(1.0), SVector{1, Float64}(3.0), false)
    pos_same3 = MC{1,NS}(1.0, 1.0, Interval(1.0, 1.0), SVector{1, Float64}(1.0), SVector{1, Float64}(1.0), false)

    @test abs(eval_check(exp_cv, pos) - exp(pos).cv) <= 1E-15
    @test abs(eval_check(exp_cc, pos) - exp(pos).cc) <= 1E-15
    @test abs(eval_check(exp_lo, pos) - exp(pos).Intv.lo) <= 1E-15
    @test abs(eval_check(exp_hi, pos) - exp(pos).Intv.hi) <= 1E-15
    
    @test abs(eval_check(exp_cv, mix) - exp(mix).cv) <= 1E-15
    @test abs(eval_check(exp_cc, mix) - exp(mix).cc) <= 1E-15
    @test abs(eval_check(exp_lo, mix) - exp(mix).Intv.lo) <= 1E-15
    @test abs(eval_check(exp_hi, mix) - exp(mix).Intv.hi) <= 1E-15

    @test abs(eval_check(exp_cv, neg) - exp(neg).cv) <= 1E-15
    @test abs(eval_check(exp_cc, neg) - exp(neg).cc) <= 1E-15
    @test abs(eval_check(exp_lo, neg) - exp(neg).Intv.lo) <= 1E-15
    @test abs(eval_check(exp_hi, neg) - exp(neg).Intv.hi) <= 1E-15
    
    @test abs(eval_check(exp_cv, pos_same1) - exp(pos_same1).cv) <= 1E-15
    @test abs(eval_check(exp_cc, pos_same1) - exp(pos_same1).cc) <= 1E-15
    @test abs(eval_check(exp_lo, pos_same1) - exp(pos_same1).Intv.lo) <= 1E-15
    @test abs(eval_check(exp_hi, pos_same1) - exp(pos_same1).Intv.hi) <= 1E-15
    
    @test abs(eval_check(exp_cv, pos_same2) - exp(pos_same2).cv) <= 1E-15
    @test abs(eval_check(exp_cc, pos_same2) - exp(pos_same2).cc) <= 1E-15
    @test abs(eval_check(exp_lo, pos_same2) - exp(pos_same2).Intv.lo) <= 1E-15
    @test abs(eval_check(exp_hi, pos_same2) - exp(pos_same2).Intv.hi) <= 1E-15
    
    @test abs(eval_check(exp_cv, pos_same3) - exp(pos_same3).cv) <= 1E-15
    @test abs(eval_check(exp_cc, pos_same3) - exp(pos_same3).cc) <= 1E-15
    @test abs(eval_check(exp_lo, pos_same3) - exp(pos_same3).Intv.lo) <= 1E-15
    @test abs(eval_check(exp_hi, pos_same3) - exp(pos_same3).Intv.hi) <= 1E-15
end
