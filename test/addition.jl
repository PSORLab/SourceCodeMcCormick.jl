
@testset "Addition" begin
    @variables x, y

    to_compute = y+5
    posreal_add_cv, posreal_add_cc, posreal_add_lo, posreal_add_hi, posreal_order = all_evaluators(to_compute)

    to_compute = y-5
    negreal_add_cv, negreal_add_cc, negreal_add_lo, negreal_add_hi, negreal_order = all_evaluators(to_compute)

    to_compute = x+y
    add_cv, add_cc, add_lo, add_hi, add_order = all_evaluators(to_compute)

    # Addition rules are very simple; each component of the McCormick expansion
    # is added separately. No need to test more than one type of McCormick object
    # from McCormick.jl
    y_1D = MC{1,NS}(1.0, 2.0, Interval(0.0, 3.0), SVector{1, Float64}(0.5), SVector{1,Float64}(2.5), false)

    xMC = MC{2,NS}(1.0, 1.5, Interval(0.5, 2.0), SVector{2, Float64}(1.0, 2.0), SVector{2, Float64}(3.0, 4.0), false)
    yMC = MC{2,NS}(-0.5, 0.5, Interval(-2.0, 2.0), SVector{2, Float64}(-0.5, 0.5), SVector{2, Float64}(1.0, -1.0), false)
    
    @test abs(eval_check(posreal_add_cv, y_1D) - (y_1D+5).cv) <= 1E-15
    @test abs(eval_check(posreal_add_cc, y_1D) - (y_1D+5).cc) <= 1E-15
    @test abs(eval_check(posreal_add_lo, y_1D) - (y_1D+5).Intv.lo) <= 1E-15
    @test abs(eval_check(posreal_add_hi, y_1D) - (y_1D+5).Intv.hi) <= 1E-15

    @test abs(eval_check(negreal_add_cv, y_1D) - (y_1D-5).cv) <= 1E-15
    @test abs(eval_check(negreal_add_cc, y_1D) - (y_1D-5).cc) <= 1E-15
    @test abs(eval_check(negreal_add_lo, y_1D) - (y_1D-5).Intv.lo) <= 1E-15
    @test abs(eval_check(negreal_add_hi, y_1D) - (y_1D-5).Intv.hi) <= 1E-15
    
    @test abs(eval_check(add_cv, xMC, yMC) - (xMC+yMC).cv) <= 1E-15
    @test abs(eval_check(add_cc, xMC, yMC) - (xMC+yMC).cc) <= 1E-15
    @test abs(eval_check(add_lo, xMC, yMC) - (xMC+yMC).Intv.lo) <= 1E-15
    @test abs(eval_check(add_hi, xMC, yMC) - (xMC+yMC).Intv.hi) <= 1E-15
end
