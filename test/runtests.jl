using Test, SourceCodeMcCormick, Symbolics, McCormick

function eval_check(eval_func, MC1, MC2)
    return eval_func(MC1.cc, MC1.cv, MC1.Intv.hi, 
            MC1.Intv.lo, MC2.cc, MC2.cv, MC2.Intv.hi, MC2.Intv.lo)
end


@testset "Multiplication" begin
    @variables x, y
    to_compute = x*y
    mult_lo, mult_hi, mult_cv, mult_cc, order = all_evaluators(to_compute)

    xMC = MC{1,NS}(-2.0, Interval(-3.0, -1.0), 1)
    yMC = MC{1,NS}(4.0, Interval(2.0, 6.0), 2)
    zMC = 0.5*xMC*yMC
    neg = zMC
    mix = zMC + 4.0
    pos = zMC + 10.0
    
    @test eval_check(mult_lo, neg, neg) == (neg*neg).Intv.lo
    @test eval_check(mult_lo, neg, mix) == (neg*mix).Intv.lo
    @test eval_check(mult_lo, neg, pos) == (neg*pos).Intv.lo
    @test eval_check(mult_lo, mix, neg) == (mix*neg).Intv.lo
    @test eval_check(mult_lo, mix, mix) == (mix*mix).Intv.lo
    @test eval_check(mult_lo, mix, pos) == (mix*pos).Intv.lo
    @test eval_check(mult_lo, pos, neg) == (pos*neg).Intv.lo
    @test eval_check(mult_lo, pos, mix) == (pos*mix).Intv.lo
    @test eval_check(mult_lo, pos, pos) == (pos*pos).Intv.lo

    @test eval_check(mult_hi, neg, neg) == (neg*neg).Intv.hi
    @test eval_check(mult_hi, neg, mix) == (neg*mix).Intv.hi
    @test eval_check(mult_hi, neg, pos) == (neg*pos).Intv.hi
    @test eval_check(mult_hi, mix, neg) == (mix*neg).Intv.hi
    @test eval_check(mult_hi, mix, mix) == (mix*mix).Intv.hi
    @test eval_check(mult_hi, mix, pos) == (mix*pos).Intv.hi
    @test eval_check(mult_hi, pos, neg) == (pos*neg).Intv.hi
    @test eval_check(mult_hi, pos, mix) == (pos*mix).Intv.hi
    @test eval_check(mult_hi, pos, pos) == (pos*pos).Intv.hi

    @test eval_check(mult_cv, neg, neg) == (neg*neg).cv
    @test eval_check(mult_cv, neg, mix) == (neg*mix).cv
    @test eval_check(mult_cv, neg, pos) == (neg*pos).cv
    @test eval_check(mult_cv, mix, neg) == (mix*neg).cv
    @test eval_check(mult_cv, mix, mix) == (mix*mix).cv
    @test eval_check(mult_cv, mix, pos) == (mix*pos).cv
    @test eval_check(mult_cv, pos, neg) == (pos*neg).cv
    @test eval_check(mult_cv, pos, mix) == (pos*mix).cv
    @test eval_check(mult_cv, pos, pos) == (pos*pos).cv

    @test eval_check(mult_cc, neg, neg) == (neg*neg).cc
    @test eval_check(mult_cc, neg, mix) == (neg*mix).cc
    @test eval_check(mult_cc, neg, pos) == (neg*pos).cc
    @test eval_check(mult_cc, mix, neg) == (mix*neg).cc
    @test eval_check(mult_cc, mix, mix) == (mix*mix).cc
    @test eval_check(mult_cc, mix, pos) == (mix*pos).cc
    @test eval_check(mult_cc, pos, neg) == (pos*neg).cc
    @test eval_check(mult_cc, pos, mix) == (pos*mix).cc
    @test eval_check(mult_cc, pos, pos) == (pos*pos).cc
end

@testset "Addition" begin
    @variables x, y
    to_compute = x+y
    add_lo, add_hi, add_cv, add_cc, order = all_evaluators(to_compute)

    xMC = MC{1,NS}(-2.0, Interval(-3.0, -1.0), 1)
    yMC = MC{1,NS}(4.0, Interval(2.0, 6.0), 2)
    zMC = 0.5*xMC*yMC
    neg = zMC
    mix = zMC + 4.0
    pos = zMC + 10.0
    
    @test eval_check(add_lo, neg, neg) == (neg+neg).Intv.lo
    @test eval_check(add_lo, neg, mix) == (neg+mix).Intv.lo
    @test eval_check(add_lo, neg, pos) == (neg+pos).Intv.lo
    @test eval_check(add_lo, mix, neg) == (mix+neg).Intv.lo
    @test eval_check(add_lo, mix, mix) == (mix+mix).Intv.lo
    @test eval_check(add_lo, mix, pos) == (mix+pos).Intv.lo
    @test eval_check(add_lo, pos, neg) == (pos+neg).Intv.lo
    @test eval_check(add_lo, pos, mix) == (pos+mix).Intv.lo
    @test eval_check(add_lo, pos, pos) == (pos+pos).Intv.lo

    @test eval_check(add_hi, neg, neg) == (neg+neg).Intv.hi
    @test eval_check(add_hi, neg, mix) == (neg+mix).Intv.hi
    @test eval_check(add_hi, neg, pos) == (neg+pos).Intv.hi
    @test eval_check(add_hi, mix, neg) == (mix+neg).Intv.hi
    @test eval_check(add_hi, mix, mix) == (mix+mix).Intv.hi
    @test eval_check(add_hi, mix, pos) == (mix+pos).Intv.hi
    @test eval_check(add_hi, pos, neg) == (pos+neg).Intv.hi
    @test eval_check(add_hi, pos, mix) == (pos+mix).Intv.hi
    @test eval_check(add_hi, pos, pos) == (pos+pos).Intv.hi

    @test eval_check(add_cv, neg, neg) == (neg+neg).cv
    @test eval_check(add_cv, neg, mix) == (neg+mix).cv
    @test eval_check(add_cv, neg, pos) == (neg+pos).cv
    @test eval_check(add_cv, mix, neg) == (mix+neg).cv
    @test eval_check(add_cv, mix, mix) == (mix+mix).cv
    @test eval_check(add_cv, mix, pos) == (mix+pos).cv
    @test eval_check(add_cv, pos, neg) == (pos+neg).cv
    @test eval_check(add_cv, pos, mix) == (pos+mix).cv
    @test eval_check(add_cv, pos, pos) == (pos+pos).cv

    @test eval_check(add_cc, neg, neg) == (neg+neg).cc
    @test eval_check(add_cc, neg, mix) == (neg+mix).cc
    @test eval_check(add_cc, neg, pos) == (neg+pos).cc
    @test eval_check(add_cc, mix, neg) == (mix+neg).cc
    @test eval_check(add_cc, mix, mix) == (mix+mix).cc
    @test eval_check(add_cc, mix, pos) == (mix+pos).cc
    @test eval_check(add_cc, pos, neg) == (pos+neg).cc
    @test eval_check(add_cc, pos, mix) == (pos+mix).cc
    @test eval_check(add_cc, pos, pos) == (pos+pos).cc
end


@testset "Subtraction" begin
    @variables x, y
    to_compute = x-y
    sub_lo, sub_hi, sub_cv, sub_cc, order = all_evaluators(to_compute)

    xMC = MC{1,NS}(-2.0, Interval(-3.0, -1.0), 1)
    yMC = MC{1,NS}(4.0, Interval(2.0, 6.0), 2)
    zMC = 0.5*xMC*yMC
    neg = zMC
    mix = zMC + 4.0
    pos = zMC + 10.0
    
    @test eval_check(sub_lo, neg, neg) == (neg-neg).Intv.lo
    @test eval_check(sub_lo, neg, mix) == (neg-mix).Intv.lo
    @test eval_check(sub_lo, neg, pos) == (neg-pos).Intv.lo
    @test eval_check(sub_lo, mix, neg) == (mix-neg).Intv.lo
    @test eval_check(sub_lo, mix, mix) == (mix-mix).Intv.lo
    @test eval_check(sub_lo, mix, pos) == (mix-pos).Intv.lo
    @test eval_check(sub_lo, pos, neg) == (pos-neg).Intv.lo
    @test eval_check(sub_lo, pos, mix) == (pos-mix).Intv.lo
    @test eval_check(sub_lo, pos, pos) == (pos-pos).Intv.lo

    @test eval_check(sub_hi, neg, neg) == (neg-neg).Intv.hi
    @test eval_check(sub_hi, neg, mix) == (neg-mix).Intv.hi
    @test eval_check(sub_hi, neg, pos) == (neg-pos).Intv.hi
    @test eval_check(sub_hi, mix, neg) == (mix-neg).Intv.hi
    @test eval_check(sub_hi, mix, mix) == (mix-mix).Intv.hi
    @test eval_check(sub_hi, mix, pos) == (mix-pos).Intv.hi
    @test eval_check(sub_hi, pos, neg) == (pos-neg).Intv.hi
    @test eval_check(sub_hi, pos, mix) == (pos-mix).Intv.hi
    @test eval_check(sub_hi, pos, pos) == (pos-pos).Intv.hi

    @test eval_check(sub_cv, neg, neg) == (neg-neg).cv
    @test eval_check(sub_cv, neg, mix) == (neg-mix).cv
    @test eval_check(sub_cv, neg, pos) == (neg-pos).cv
    @test eval_check(sub_cv, mix, neg) == (mix-neg).cv
    @test eval_check(sub_cv, mix, mix) == (mix-mix).cv
    @test eval_check(sub_cv, mix, pos) == (mix-pos).cv
    @test eval_check(sub_cv, pos, neg) == (pos-neg).cv
    @test eval_check(sub_cv, pos, mix) == (pos-mix).cv
    @test eval_check(sub_cv, pos, pos) == (pos-pos).cv

    @test eval_check(sub_cc, neg, neg) == (neg-neg).cc
    @test eval_check(sub_cc, neg, mix) == (neg-mix).cc
    @test eval_check(sub_cc, neg, pos) == (neg-pos).cc
    @test eval_check(sub_cc, mix, neg) == (mix-neg).cc
    @test eval_check(sub_cc, mix, mix) == (mix-mix).cc
    @test eval_check(sub_cc, mix, pos) == (mix-pos).cc
    @test eval_check(sub_cc, pos, neg) == (pos-neg).cc
    @test eval_check(sub_cc, pos, mix) == (pos-mix).cc
    @test eval_check(sub_cc, pos, pos) == (pos-pos).cc
end
