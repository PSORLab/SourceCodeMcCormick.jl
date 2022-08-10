# SourceCodeMcCormick.jl

This package is an experimental approach to use source-code transformation to apply McCormick relaxations
to symbolic functions for use in deterministic global optimization. While packages like `McCormick.jl`
take set-valued McCormick objects and utilize McCormick relaxation rules to overload standard math operations,
`SourceCodeMcCormick.jl` aims to interpret symbolic expressions, apply McCormick transformations, and
return a new symbolic expression representing those relaxations. This functionality is designed to be
used for both algebraic and dynamic systems.

Experimental Approach to McCormick Relaxation Source-Code Transformation for Differential Inequalities

## Algebraic Systems

For a given algebraic equation or system of equations, `SourceCodeMcCormick` is designed to provide
symbolic transformations that represent the lower/upper bounds and convex/concave relaxations of the
provided equation(s). Most notably, `SourceCodeMcCormick` uses this symbolic transformation to
generate "evaluation functions" which, for a given expression, return the lower/upper bounds and
convex/concave relaxations of an expression. E.g.:

```
using SourceCodeMcCormick, Symbolics

@variables x
to_compute = x*(15+x)^2
x_lo_eval, x_hi_eval, x_cv_eval, x_cc_eval, order = all_evaluators(to_compute)
```

Here, the outputs marked `_eval` are the evaluation functions for the lower bound (`lo`), upper
bound (`hi`), convex underestimator (`cv`), and concave overestimator (`cc`) of the `to_compute`
expression. The inputs to each of these functions are described by the `order` vector, which 
in this case is `[x_cc, x_cv, x_hi, x_lo]`. 

There are several important benefits of using these functions. First, they are fast. Normally,
applying McCormick relaxations takes some time as the relaxation depends on the provided bounds
and relaxation values of the input(s), but also on the form of the expression itself. Packages 
like `McCormick.jl` use control flow to quickly determine the form of the relaxations and provide
McCormick objects of the results, but because the forms of the expressions are not known *a priori*,
this control flow takes time to process. In contrast, `SourceCodeMcCormick` effectively hard-codes
the relaxations for a specific, pre-defined function into the evaluation functions, making it
inherently faster. For example:

```
using BenchmarkTools, McCormick

xMC = MC{1, NS}(2.5, Interval(-1.0, 4.0), 1)

@btime xMC*(15+xMC)^2
# 228.381 ns (15 allocations: 384 bytes)
# MC{1, NS}(683.5, 952.0, [-361, 1444], [501.0], [328.0], false)

@btime x_cv_eval(2.5, 2.5, 4.0, -1.0)
# 30.382 ns (1 allocation: 16 bytes)
# 683.5
```

This is not an *entirely* fair example because the operation with xMC is simultaneously calculating
lower/upper bounds, both relaxations, and (sub)gradients of the convex/concave relaxations, but the
`SourceCodeMcCormick` result is just under an order of magnitude faster. As the expression inceases
in complexity, this speedup becomes even more apparent.

Second, these evaluation functions are compatible with `CUDA.jl` in that they are broadcastable
over `CuArray`s. Depending on the GPU, number of evaluations, and complexity of the function,
this can dramatically decrease the time to compute the numerical values of function bounds and
relaxations. E.g.:

```
using CUDA

# Using McCormick.jl
xMC_array = MC{1,NS}.(rand(10000), Interval.(zeros(10000), ones(10000)), ones(Int, 10000))
@btime xMC_array.*(15 .+ xMC_array).^2
# 1.616 ms (120012 allocations: 2.37 MiB)

# Using SourceCodeMcCormick.jl, broadcast using CPU
xcc = rand(10000)
xcv = copy(xcc)
xhi = ones(10000)
xlo = zeros(10000)
@btime x_cv_eval.(xcc, xcv, xhi, xlo)
# 100.100 μs (4 allocations: 78.27 KiB)

# Using SourceCodeMcCormick.jl and CUDA.jl, broadcast using GPU
xcc_GPU = cu(xcc)
xcv_GPU = cu(xcv)
xhi_GPU = cu(xhi)
xlo_GPU = cu(xlo)
@btime x_cv_eval.(xcc_GPU, xcv_GPU, xhi_GPU, xlo_GPU)
# 6.575 μs (33 allocations: 2.34 KiB)
```


## Dynamic Systems

For dynamic systems, `SourceCodeMcCormick` was built with a differential inequalities
approach in mind where the relaxations of derivatives are calculated in advance and
the resulting (larger) differential equation system, with explicit definitions of
the relaxations of derivatives, can be solved. For algebraic systems, the main
product of this package is the broadcastable evaluation functions. For dynamic
systems, this package follows the same idea as in algebraic systems but stops at
the symbolic representations of relaxations. This functionality is designed to work
with a `ModelingToolkit`-type `ODESystem` with factorable equations--`SourceCodeMcCormick`
will take such a system and return a new `ODESystem` with expanded equations to
provide interval extensions and (if desired) McCormick relaxations. E.g.:

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

and generates a new ODE system (`new_syst`) with equations:
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

where `x_lo < x_cv < x < x_cc < x_hi`. Only addition is shown in this example, as other operations
can appear very expansive, but the same operations available for algebraic systems are available
for dynamic systems as well. As with the algebraic evaluation functions, equations created by
`SourceCodeMcCormick` are GPU-ready--multiple trajectories of the resulting ODE system at 
different points and with different state/parameter bounds can be solved simultaneously using
an `EnsembleProblem` in the SciML ecosystem, and GPU hardware can be applied for these solves
using `DiffEqGPU.jl`.

## Limitations

Currently, as proof-of-concept, `SourceCodeMcCormick` can only handle functions with 
addition (+), subtraction (-), multiplication (\*), powers of 2 (^2), natural base
exponentials (exp), and minimum/maximum (min/max) expressions. Future work will include
adding other operations found in `McCormick.jl`. 

