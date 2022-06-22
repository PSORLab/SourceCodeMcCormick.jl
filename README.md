# SourceCodeMcCormick.jl
Experimental Approach to McCormick Relaxation Source-Code Transformation for Differential Inequalities

The purpose of this package is to transform a `ModelingToolkit` ODE system with factorable equations into a new `ModelingToolkit` ODE system with interval or McCormick relaxations applied. E.g.:

```
using SourceCodeMcCormick, ModelingToolkit
@parameters p[1:2] t
@variables x[1:2](t)
D = Differential(t)

tspan = (0.0, 35.0)
x0 = [1.0; 0.0]
x_dict = Dict(x[i] .=> x0[i] for i in 1:2)
p_start = [0.020; 0.025]
p_dict = Dict(p[i] .=> p_start[i] for i in 1:2)

eqns = [D(x[1]) ~ p[1]+x[1],
        D(x[2]) ~ p[2]+x[2]]

@named syst = ODESystem(eqns, t, x, p, defaults=merge(x_dict, p_dict))
new_syst = apply_transform(McCormickIntervalTransform(), syst)
```

This takes the original ODE system (`syst`) with equations:
```
Differential(t)(x[1](t)) ~ x[1](t) + p[1]
Differential(t)(x[2](t)) ~ x[2](t) + p[2]
```

and generates a new ODE system (`new_syst') with equations:
```
Differential(t)(x_1_lo(t)) ~ p_1_lo + x_1_lo(t)
Differential(t)(x_1_hi(t)) ~ p_1_hi + x_1_hi(t)
Differential(t)(x_1_cv(t)) ~ p_1_cv + x_1_cv(t)
Differential(t)(x_1_cc(t)) ~ p_1_cc + x_1_cc(t)
Differential(t)(x_2_lo(t)) ~ p_2_lo + x_2_lo(t)
Differential(t)(x_2_hi(t)) ~ p_2_hi + x_2_hi(t)
Differential(t)(x_2_cv(t)) ~ p_2_cv + x_2_cv(t)
Differential(t)(x_2_cc(t)) ~ p_2_cc + x_2_cc(t)
```

where `x_lo < x_cv < x < x_cc < x_hi`. This new system of ODEs is generated using GPU-compatible language--i.e., any decision points in the form of the resulting equation based on some terms being positive or negative are handled by IfElse.ifelse statements and/or min/max evaluations. By using only these types of expressions, multiple trajectories of the resulting ODE system can be solved simultaneously on a GPU, such as by using `DiffEqGPU` in the SciML ecosystem.