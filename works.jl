
# This case WORKED at one point, so it should work whenever you're testing something.
using ModelingToolkit, OrdinaryDiffEq
using DifferentialEquations: solve
using DiffEqGPU
using BenchmarkTools
@parameters p[1:2] t
@variables x[1:2](t)
D = Differential(t)
x0 = [1.0; 0.0]
tspan = (0.0, 35.0)
p_start = [0.020; 0.025]
eqns = [D(x[1]) ~ -p[1]*x[1] + p[2]*x[2],
        D(x[2]) ~ p[1]*x[1] - p[2]*x[2]]

@named test_works = ODESystem(eqns, t, x, p, default_u0=Dict( x[i] .=> x0[i] for i in 1:2), default_p = Dict( p[i] .=> p_start[i] for i in 1:2))
tested_MC_works = apply_transform(McCormickIntervalTransform(), test_works) #This one breaks
set_bounds(tested_MC_works, p[1], (0.015, 0.025))
ode_problem_works = eval(ODEProblemExpr(structural_simplify(tested_MC_works), tested_MC_works.defaults, tspan))

# Can create u's and p's to interpolate into the prob_func definition like this:
p_order = tested_MC_works.ps.value
u_order = structural_simplify(tested_MC_works).states.value
u_list = []
p_list = []
for i = 1:11
    push!(u_list, [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0+(i-1)*0.1])
    push!(p_list, [0.01, 0.01, 0.01, 0.01, 0.01+(i-1)*0.001, 0.01-(i-1)*0.001, 0.01, 0.01])
end
prob_func_works = (prob, i, repeat) -> remake(prob, u0=:($u_list)[i], p = :($p_list)[i])

# And this ensemble problem works with EnsembleGPUArray, currently.
ensemble_works = EnsembleProblem(ode_problem_works, prob_func=prob_func_works)
@benchmark solve(ensemble_works, Tsit5(), EnsembleGPUArray(), trajectories=11)