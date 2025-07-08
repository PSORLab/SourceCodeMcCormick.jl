# SourceCodeMcCormick.jl

| **PSOR Lab** | **Build Status**                                                                                |
|:------------:|:-----------------------------------------------------------------------------------------------:|
| [![](https://img.shields.io/badge/Developed_by-PSOR_Lab-342674)](https://psor.uconn.edu/) | [![Build Status](https://github.com/PSORLab/SourceCodeMcCormick.jl/workflows/CI/badge.svg?branch=master)](https://github.com/PSORLab/SourceCodeMcCormick.jl/actions?query=workflow%3ACI) [![codecov](https://codecov.io/gh/PSORLab/SourceCodeMcCormick.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/PSORLab/SourceCodeMcCormick.jl)|


This package uses source-code generation to create CUDA kernels that return evaluations of McCormick-based
relaxations. Expressions composed of `Symbolics.jl`-type variables can be passed into the main `SourceCodeMcCormick.jl`
(`SCMC`) function, `kgen`, after which the expressions are factored, algebraic rearrangements and other
modifications are applied as necessary, and a new, custom function is written that calculates inclusion
monotonic interval extensions, convex and concave relaxations, and subgradients of the convex and concave
relaxations of the original expression. These new custom functions are meant to be used in, e.g., a branch-and-bound
routine where the optimizer would like to calculate relaxations for many nodes simultaneously using GPU 
resources. They are designed to be used with `CUDA.CuArray{Float64}` objects, with 64-bit (double-precision)
numbers recommended to maintain accuracy.


## Basic Functionality

The primary user-facing function is `kgen` ("kernel generator"). The `kgen` function returns a source-code-generated
function that provides evaluations of convex and concave relaxations, inclusion monotonic interval extensions,
convex relaxation subgradients, and concave relaxation subgradients, for an input symbolic expression. Inputs
to the newly generated function are expected to be an output placeholder, followed by a separate `CuArray` for
each input variable, sorted in alphabetical order. The "input" variables must be `(m,3)`-dimensional `CuArray`s,
with the columns corresponding to points where evaluations are requested, lower bounds, and upper bounds,
respectively. The "output" placeholder must be `(m,4+2n)`-dimensional, where `n` is the dimensionality of the
expression. The columns will hold, respectively, the convex and concave relaxations, lower and upper bounds of
an interval extension, a subgradient of the convex relaxation, and a subgradient of the concave relaxation. 
Each row `m` in the inputs and output are independent of one another, allowing calculations of relaxation information
at multiple points on (potentially) different domains. A demonstration of how to use `kgen` is shown here,
with the output compared to the multiple-dispatch-based `McCormick.jl`:

```julia
using SourceCodeMcCormick, Symbolics, CUDA
Symbolics.@variables x, y
expr = exp(x/y) - (x*y^2)/(y+1)
new_func = kgen(expr)
x_in = CuArray([1.0 0.5 3.0;])
y_in = CuArray([0.7 0.1 2.0;])
OUT = CUDA.zeros(Float64, 1, 8)

using McCormick
xMC = MC{2,NS}(1.0, Interval(0.5, 3.0), 1)
yMC = MC{2,NS}(0.7, Interval(0.1, 2.0), 2)


julia> new_func(OUT, x_in, y_in)  # SourceCodeMcCormick.jl
CUDA.HostKernel for f_3zSdMo7LFdg_1(CuDeviceMatrix{Float64, 1}, CuDeviceMatrix{Float64, 1}, CuDeviceMatrix{Float64, 1})

julia> Array(OUT)
1×8 Matrix{Float64}:
 0.228368  2.96348e12  -9.62507  1.06865e13  -2.32491  -3.62947  3.59209e12  -8.98023e11

julia> exp(xMC/yMC) - (xMC*yMC^2)/(yMC+1)  # McCormick.jl
MC{2, NS}(0.22836802303235793, 2.963476144457207e12, [-9.62507, 1.06865e+13], [-2.3249068975747305, -3.6294726270892967], [3.592092296310309e12, -8.980230740778097e11], false)
```

In this example, the symbolic expression `exp(x/y) - (x*y^2)/(y+1)` is passed into `kgen`, which returns the
function `new_func`. Passing inputs representing the values and domain where calculations are desired for `x` 
and `y` (i.e., evaluating `x` at `1.0` in the domain `[0.5, 3.0]` and `y` at `0.7` in the domain `[0.1, 2.0]`) 
into `new_func` returns pointwise evaluations of {cv, cc, lo, hi, [cvgrad]..., [ccgrad]...} for the original
expression of `exp(x/y) - (x*y^2)/(y+1)` on the specified domain of `x` and `y`.

Evaluating large numbers of points simultaneously using a GPU allows for faster relaxation calculations than 
evaluating points individually using a CPU. This is demonstrated in the following example:

```julia
using SourceCodeMcCormick, Symbolics, CUDA, McCormick, BenchmarkTools
Symbolics.@variables x, y
expr = exp(x/y) - (x*y^2)/(y+1)
new_func = kgen(expr)

# CuArray values for SCMC timing
x_in = hcat(2.5.*CUDA.rand(Float64, 8192) .+ 0.5, 
            0.5.*CUDA.ones(Float64, 8192), 
            3.0.*CUDA.ones(Float64, 8192))
y_in = hcat(1.9.*CUDA.rand(Float64, 8192) .+ 0.1, 
            0.1.*CUDA.ones(Float64, 8192), 
            2.0.*CUDA.ones(Float64, 8192))
OUT = CUDA.zeros(Float64, 8192, 8)

# McCormick objects for McCormick.jl timing
xMC = MC{2,NS}(1.0, Interval(0.5, 3.0), 1)
yMC = MC{2,NS}(0.7, Interval(0.1, 2.0), 2)

########################################################
##### Timing 
#   CPU: Intel i7-9850H
#   GPU: NVIDIA Quadro T2000


##### McCormick.jl (single evaluation on CPU)
@btime exp($xMC/$yMC) - ($xMC*$yMC^2)/($yMC+1);
#   236.941 ns (0 allocations: 0 bytes)


##### SourceCodeMcCormick.jl (8192 simultaneous evaluations on GPU)
@btime new_func($OUT, $x_in, $y_in);
#   75.700 μs (18 allocations: 384 bytes)

#   Average time per evaluation = 75.700 μs / 8192 = 9.241 ns


# Time to move results back to CPU (if needed):
@btime Array($OUT);
#   107.200 μs (8 allocations: 512.16 KiB)

# Time including evaluation:
# 75.700 μs + 107.200 μs = 182.900 μs 
#   Average time per evaluation = 182.900 μs / 8192 = 22.327 ns
```

As shown by the two previous examples, functions generated by `kgen` can be significantly faster than
the same calculations done by `McCormick.jl`, and provide the same information for a given input
expression. There is a considerable time penalty for bringing the results from device memory (GPU) back
to host memory (CPU) due to the volume of information generated. If possible, an algorithm designed
to make use of `SCMC`-generated functions should utilize calculated relaxation information within the
GPU rather than transfer the information between hardware components.


## Arguments for `kgen`

The `kgen` function can be called a `Num`-type expression as an input (as in the examples in the
previous section), or with several possible arguments that affect `kgen`'s behavior. Some of these
functionalities and their use cases are shown in this section.

### 1) Overwrite

Functions and kernels created by `kgen` are human-readable and are [currently] saved in the 
"src\kernel_writer\storage" folder with a title created by hashing the input expression and
list of relevant variables. By saving the files in this way, calling `kgen` multiple times
with the same expression in the same Julia session can skip the steps of re-writing and 
re-compilling the same expression. If the same expression is used across Julia sessions,
`kgen` can skip the re-writing step and instead compile the existing function and kernels.

In some cases, this behavior may not be desired. For example, if there is an error or interrupt
during the initial writing of the kernel, or if the source code of `kgen` is being adjusted
to modify how kernels are written, it is preferable to re-write the kernel from scratch
rather than rely on the existing generated code. For these cases, the setting `overwrite=true`
may be used to tell `kgen` to re-write and re-compile the generated code. For example:

```julia
using SourceCodeMcCormick, Symbolics, CUDA
Symbolics.@variables x, y
expr = exp(x/y) - (x*y^2)/(y+1)

# First call with an entirely new expression
@time new_func = kgen(expr);
#  3.958299 seconds (3.71 M allocations: 162.940 MiB, 13.60% compilation time)

# Second call in the same Julia session
@time new_func = kgen(expr);
#  0.000310 seconds (252 allocations: 10.289 KiB)

# Third call, with a hint for `kgen` to re-write and re-compile
@time new_func = kgen(expr, overwrite=true);
#  3.933316 seconds (3.70 M allocations: 162.619 MiB, 13.10% compilation time)
```


### 2) Problem Variables

When using `SCMC` to calculate relaxations for expressions that are components of a larger optimization
problem (such as individual constraints), it is necessary to provide `kgen` information about all of the
problem variables. For example, if the objective function in a problem is `x + y + z`, and one constraint
is `y^2 - z <= 0`, the subgradients of the relaxations for the constraint should be 3-dimensional despite
the constraint only depending on `y` and `z`. Further, the dimensions must match across expressions such
that the first subgradient dimension always corresponds to `x`, the second always corresponds to `y`, and
so on. This is critically important for the primary use case of `SCMC` and is built in to work without
the need to use a keyword argument. For example:

```julia
using SourceCodeMcCormick, Symbolics, CUDA
Symbolics.@variables x, y, z
obj = x + y + z
constr1 = y^2 - z

# A valid way to call `kgen`, which is incorrect in this case because the problem
# variables are {x, y, z}, but `constr1` only depends on {y, z}. 
constr1_func_A = kgen(constr1)

# The more correct way to call `kgen` in this case, to inform it that the problem 
# variables are {x, y, z}. 
constr1_func_B = kgen(constr1, [x, y, z])
```

In this example, `constr1_func_A` assumes that the only variables are `y` and `z`, and it therefore expects
the `OUT` storage to be of size `(m,8)` (i.e., each subgradient is 2-dimensional). Because `kgen` is being
used for one constraint as part of an optimization problem, it is more correct to use `constr1_func_B`,
which expects `OUT` to be of size `(m,10)` (i.e., each subgradient is 3-dimensional). Note that this
is directly analogous to the typical use of `McCormick.jl`, where `MC` objects are constructed with the
larger problem in mind. E.g.:

```julia
using McCormick

xMC = MC{3,NS}(1.0, Interval(0.0, 2.0), 1)
yMC = MC{3,NS}(1.5, Interval(1.0, 3.0), 2)
zMC = MC{3,NS}(1.8, Interval(1.0, 4.0), 3)

obj = xMC + yMC + zMC
constr1 = yMC^2 - zMC
```

Here, for the user to calculate `constr1`, they must have already created the `MC` objects `yMC` and `zMC`.
In the construction of these objects, it was specified that the subgradient would be 3-dimensional (the
`3` in `MC{3,NS}`), and `yMC` and `zMC` were selected as the second and third dimensions of the subgradient
by the `2` and `3` that follow the call to `Interval()`. `SCMC` needs the same information; only the
format is different from `McCormick.jl`. 

Note that if a list of problem variables is not provided, `SCMC` will collect the variables in the 
expression it's provided and sort them alphabetically, and then numerically. I.e., `x1 < x2 < x10 < y1`. 
If a variable list is provided, `SCMC` will respect the order of variables in the list without sorting
them. For example, if `constr1_func_B` were created by calling `kgen(constr1, [z, y, x])`, then
`constr1_func_B` would expect its input arguments to correspond to `(OUTPUT, z, y, x)` instead of
the alphabetical `(OUTPUT, x, y, z)`.


### 3) Splitting

Internally, `kgen` may split the input expression into multiple subexpressions if the generated kernels
would be too long. In general, longer kernels are faster but require longer compilation time. The default
settings attempt to provide a balance of good performance and acceptably low compilation times, but
situations may arise where a different balance is preferred. The amount of splitting can be modified by
setting the keyword argument `splitting` to one of `{:low, :default, :high, :max}`, where higher values
indicate the (internal) creation of more, shorter kernels (faster compilation, slower performance), and lower
values indicate the (internal) creation of fewer, longer kernels (longer compilation, faster performance). 
Actual compilation time and performance, as well as the impact of different `splitting` settings, is 
expression-dependent. In most cases, however, the default value should be sufficient.

Here's an example showing how different values affect compilation time and performance:
```julia
using SourceCodeMcCormick, Symbolics, CUDA
Symbolics.@variables x, y
expr = exp(x/y) - (x*y^2)/(y+1)

# CuArray values for SCMC timing
x_in = hcat(2.5.*CUDA.rand(Float64, 8192) .+ 0.5, 
            0.5.*CUDA.ones(Float64, 8192), 
            3.0.*CUDA.ones(Float64, 8192))
y_in = hcat(1.9.*CUDA.rand(Float64, 8192) .+ 0.1, 
            0.1.*CUDA.ones(Float64, 8192), 
            2.0.*CUDA.ones(Float64, 8192))
OUT = CUDA.zeros(Float64, 8192, 8)

##################################
# Default splitting (creates 1 kernel internally, based on the size of this expr)
@time new_func = kgen(expr, overwrite=true);
#  4.100798 seconds (3.70 M allocations: 162.637 MiB, 1.47% gc time, 12.29% compilation time)

@btime new_func($OUT, $x_in, $y_in);
#  77.000 μs (18 allocations: 384 bytes)

# --> This provides a good balance of compilation time and fast performance

##################################
# Higher splitting (creates 2 kernels internally)
@time new_func = kgen(expr, splitting=:high, overwrite=true);
#  3.452594 seconds (3.87 M allocations: 170.845 MiB, 13.87% compilation time)

@btime new_func($OUT, $x_in, $y_in);
#  117.000 μs (44 allocations: 960 bytes)

# --> Splitting the function into 2 shorter kernels yields a marginally faster
#     compilation time, but performance is marginally worse due to needing twice
#     as many calls to kernels

##################################
# Maximum splitting (creates 4 kernels internally)
@time new_func = kgen(expr, splitting=:max, overwrite=true);
#  5.903894 seconds (5.88 M allocations: 260.864 MiB, 12.91% compilation time)

@btime new_func($OUT, $x_in, $y_in);
#  184.900 μs (83 allocations: 1.91 KiB)

# --> Given the simplicity of the expression, splitting into 4 kernels does not
#     provide benefit over the 2 kernels created with :high. Compilation time
#     is longer because twice as many kernels need to be compiled (and each is
#     not much less complicated than in the :high case), and performance is worse
#     (because the number of kernel calls is doubled relative to the :high case
#     without much decrease in complexity). 
```

## The ParBB Algorithm

The intended use of `SCMC` is in conjunction with a branch-and-bound algorithm that is able 
to make use of relaxations and subgradient information that are calculated in parallel on a 
GPU. An implementation of such an algorithm is available for `SCMC` version 0.4, and is
under development for version 0.5. See the paper reference in the section "Citing SourceCodeMcCormick" for 
a more complete description of the ParBB algorithm for version 0.4. Briefly, ParBB is built 
as an extension of the EAGO solver, and it works by parallelizing the node procesing routines 
in such a way that tasks may be performed by a GPU.


## Citing SourceCodeMcCormick

Please cite the following paper when using SourceCodeMcCormick.jl. In plain text form this is:
```
Gottlieb, R. X., Xu, P., and Stuber, M. D. Automatic source code generation for deterministic
global optimization with parallel architectures. Optimization Methods and Software, 1–39 (2024).
DOI: 10.1080/10556788.2024.2396297
```
A BibTeX entry is given below:
```bibtex
@Article{doi:10.1080/10556788.2024.2396297,
  author    = {Robert X. Gottlieb, Pengfei Xu, and Matthew D. Stuber},
  journal   = {Optimization Methods and Software},
  title     = {Automatic source code generation for deterministic global optimization with parallel architectures},
  year      = {2024},
  pages     = {1--39},
  doi       = {10.1080/10556788.2024.2396297},
  eprint    = {https://doi.org/10.1080/10556788.2024.2396297},
  publisher = {Taylor \& Francis},
  url       = {https://doi.org/10.1080/10556788.2024.2396297},
}
```


# README for v0.4 (Deprecated)

This package uses source-code transformation to construct McCormick-based relaxations. Expressions composed
of `Symbolics.jl`-type variables can be passed into `SourceCodeMcCormick.jl` (`SCMC`) functions, after which
the expressions are factored, generalized McCormick relaxation rules and inclusion monotonic interval
extensions are applied to the factors, and the factors are recombined symbolically to create expressions
representing inclusion monotonic interval extensions, convex and concave relaxations, and subgradients of
convex and concave relaxations of the original expression. The new expressions are compiled into functions
that return pointwise values of these elements, which can be used in, e.g., a branch-and-bound routine. 
These functions can be used with floating-point values, vectors of floating-point values, or CUDA arrays 
of floating point values (using `CUDA.jl`) to return outputs of the same type. 64-bit (double-precision)
numbers are recommended
for relaxations to maintain accuracy.


## Basic Functionality (v0.4)

The primary user-facing function is `fgen` ("function generator"). The `fgen` function returns a source-code-generated
function that provides evaluations of convex and concave relaxations, inclusion monotonic interval extensions,
convex relaxation subgradients, and concave relaxation subgradients, for an input symbolic expression. Inputs
to the newly generated function are [currently] output to the REPL, and are sorted primarily by variable name
and secondarily by the order {cv, cc, lo, hi}. E.g., if variables `x` and `y` are used, the input to the generated
function will be `{x_cv, x_cc, x_lo, x_hi, y_cv, y_cc, y_lo, y_hi}`. A demonstration of how to use `fgen` is shown
here, with the output compared to the multiple-dispatch-based `McCormick.jl`:

```julia
using SourceCodeMcCormick, Symbolics
Symbolics.@variables x, y
expr = exp(x/y) - (x*y^2)/(y+1)
new_func = fgen(expr)
xcv, xcc, xlo, xhi = 1.0, 1.0, 0.5, 3.0
ycv, ycc, ylo, yhi = 0.7, 0.7, 0.1, 2.0

using McCormick
xMC = MC{2,NS}(1.0, Interval(0.5, 3.0), 1)
yMC = MC{2,NS}(0.7, Interval(0.1, 2.0), 2)


julia> new_func(xcv, xcc, xlo, xhi, ycv, ycc, ylo, yhi)  # SourceCodeMcCormick.jl
(0.22836802303235837, 2.963476144457207e12, -9.625065492403166, 1.068647458152446e13, -2.32490689757473, -3.629472627089296, 3.592092296310309e12, -8.980230740778097e11)

julia> exp(xMC/yMC) - (xMC*yMC^2)/(yMC+1)  # McCormick.jl
MC{2, NS}(0.22836802303235793, 2.963476144457207e12, [-9.62507, 1.06865e+13], [-2.3249068975747305, -3.6294726270892967], [3.592092296310309e12, -8.980230740778097e11], false)
```

In this example, the symbolic expression `exp(x/y) - (x*y^2)/(y+1)` is passed into `fgen`, which returns the
function `new_func`. Passing inputs representing the McCormick tuples associated with `x` and `y` (i.e., `cv` 
and `cc` being the point to evaluate for that variable, and `lo` and `hi` being the variable bounds) into 
`new_func` returns pointwise evaluations of {cv, cc, lo, hi, [cvgrad]..., [ccgrad]...} for the original
expression of `exp(x/y) - (x*y^2)/(y+1)` on the specified domain of `x` and `y`.

Functions generated using `fgen` are also built to be compatible with vectors and `CuArray`s. An example
demonstrating this capability is shown here:
```julia
using SourceCodeMcCormick, Symbolics, CUDA
@variables x
expr = (x-0.5)^2+x
new_func = fgen(expr)

xcv_GPU = CUDA.rand(Float64, 1000000)
xcc_GPU = copy(xcv_GPU)
xlo_GPU = CUDA.zeros(Float64, 1000000)
xhi_GPU = CUDA.ones(Float64, 1000000)

outputs = new_func(xcv_GPU, xcc_GPU, xlo_GPU, xhi_GPU)
```

This example shows how the generated functions can be used with CUDA arrays. The outputs generated in this
example are in the same format as for floating-point inputs (`{cv, cc, lo, hi, [cvgrad]..., [ccgrad]...}`),
but instead of the outputs being floating-point values, they are each vectors of the same length as the inputs.
This capability is useful when you want to evaluate relaxation or interval information for a large number
of points, or points on different domains, simultaneously. In this example, the domain of `x` is left
unchanged for all points for simplicity, but this is not a requirement: points on any domain can be passed
within the same function evaluation; all points are evaluated independently.

Evaluating large numbers of points simultaneously, and particularly using a GPU, allows for faster relaxation
calculations than performing individual calculations using a CPU. This is demonstrated in the following
example:

```julia
using SourceCodeMcCormick, Symbolics, McCormick, CUDA, BenchmarkTools
Symbolics.@variables x, y
expr = exp(x/y) - (x*y^2)/(y+1)
new_func = fgen(expr)

# CuArray values for SCMC timing
xcv_GPU = CuArray(2.5*rand(1000000) .+ 0.5)
xcc_GPU = copy(xcv_GPU)
xlo_GPU = 0.5 .* CUDA.ones(Float64, 1000000)
xhi_GPU = 3.0 .* CUDA.ones(Float64, 1000000)
ycv_GPU = CuArray(1.9*rand(1000000) .+ 0.1)
ycc_GPU = copy(ycv_GPU)
ylo_GPU = 0.1 .* CUDA.ones(Float64, 1000000)
yhi_GPU = 2.0 .* CUDA.ones(Float64, 1000000)

# McCormick objects for McCormick.jl timing
xMC = MC{2,NS}(1.0, Interval(0.5, 3.0), 1)
yMC = MC{2,NS}(0.7, Interval(0.1, 2.0), 2)

########################################################
##### Timing 
#   CPU: Intel i7-9850H
#   GPU: NVIDIA Quadro T2000


##### McCormick.jl
@btime exp($xMC/$yMC) - ($xMC*$yMC^2)/($yMC+1);
#   258.457 ns (0 allocations: 0 bytes)


##### SourceCodeMcCormick.jl (using GPU)

# (Outputs still on GPU)
@btime CUDA.@sync new_func($xcv_GPU, $xcc_GPU, $xlo_GPU, $xhi_GPU, $ycv_GPU, $ycc_GPU, $ylo_GPU, $yhi_GPU);
#   48.756 ms (12795 allocations: 252.33 KiB)

#   Average time per evaluation = 48.756 ms / 1000000 = 48.756 ns
#   Note: Output is `NTuple{8, CuArray{Float64, 1, CUDA.DeviceMemory}}`


# (Outputs moved to CPU memory after calculation)
@btime CUDA.@sync Array.(new_func($xcv_GPU, $xcc_GPU, $xlo_GPU, $xhi_GPU, $ycv_GPU, $ycc_GPU, $ylo_GPU, $yhi_GPU));
#   68.897 ms (12853 allocations: 61.28 MiB)

#   Average time per evaluation = 68.897 ms / 1000000 = 68.897 ns
#   Note: Output is `NTuple{8, Vector{Float64}}`
```

As shown in the example, the functions generated by `fgen` can be very fast when used with `CuArray`s, 
and as shown in the previous example, they can provide the same information as McCormick.jl for a given
input function. It is worth noting that if `CuArray`s are passed as inputs, the outputs of the generated 
function are also of type `CuArray`. I.e., if data stored in GPU memory is passed as inputs, the calculations
occur on the GPU, and the outputs are also stored in GPU memory. This is most useful if you can make use 
of the results of the relaxation calculations directly on the GPU. If you must move the results
back into CPU memory, this incurs a large time cost (as in the example: moving the results adds roughly 40%
to the total time). An active research project is focused on improvements to the underlying
SourceCodeMcCormick functionality that creates these functions which will even further improve the
calculation speed.


## Arguments for `fgen` (v0.4)

The `fgen` function can be called with only a `Num`-type expression as an input (as in the examples in the
previous section), or with many other possible arguments that affect `fgen`'s behavior. Some of these
functionalities and their use cases are shown in this section.

### 1) Constants

For any call to `fgen`, the keyword argument `constants` may be used to specify that a Symbolic variable
is not meant to be treated as a McCormick object but rather as an adjustable parameter that will always
take on a specific, constant value. This affects how the symbol is treated by SourceCodeMcCormick internally,
and will also affect how the symbol is input to the generated function.

```julia
using SourceCodeMcCormick, Symbolics
Symbolics.@variables x, y
expr = exp(x/y) - (x*y^2)/(y+1)

# Standard way of calling fgen:
new_func = fgen(expr)
xcv, xcc, xlo, xhi = 1.0, 1.0, 0.5, 3.0
ycv, ycc, ylo, yhi = 0.7, 0.7, 0.1, 2.0

outputs = new_func(xcv, xcc, xlo, xhi, ycv, ycc, ylo, yhi)
#    Note that the order of inputs is {x, y}, where each is 
#    split into a McCormick tuple of {cv, cc, lo, hi}

# Treating "y" as a constant or adjustable parameter
new_func = fgen(expr, constants=[y])
y_fixed_val = 0.7

outputs = new_func(y_fixed_val, xcv, xcc, xlo, xhi)
#    Note that the order of the inputs is adjusted so that 
#    the constants come first. I.e., the ordering goes:
#    {{constants}, {McCormick variables}} = {y, x}
#    and then the McCormick variables are split into
#    their McCormick tuples, giving: {y, x_cv, x_cc, x_lo, x_hi}
```

Marking variables as constants makes the calculations simpler, and eliminates the need to give expanded
McCormick tuples of symbols that will always represent a single value.


### 2) Specifying an Output Subset

In some cases, you may not want or need the full list of outputs `{cv, cc, lo, hi, [cvgrad]..., [ccgrad]...}`. 
E.g., if the generated function is being used in a subgradient-free lower-bounding routine within a
branch-and-bound algorithm, you may only want `{cv}` or `{cv, lo}` as the outputs. Reducing the required
outputs allows you to save space by not allocating unnecessary vectors or `CuArray`s, and it also
speeds up the overall computation of the generated function by not performing unnecessary calculations.
E.g., if the concave relaxation of the output is not required, the generated function will not calculate
the concave relaxation in the final calculation step (though calculating the concave relaxation in
intermediate steps may still be required to implement relaxation rules). Additionally, if no subgradient
information is requested, the generated function can skip all subgradient calculations, since they do
not impact the relaxation or inclusion monotonic interval calculations. Specifying the outputs can be
done as follows:

```julia
using SourceCodeMcCormick, Symbolics
Symbolics.@variables x, y
expr = exp(x/y) - (x*y^2)/(y+1)

# When fgen is called, any combination of the following outputs can be used:
#   Symbol  |  Interpretation 
# -----------------------------------------
#      :cv  |  Convex relaxation
#      :cc  |  Concave relaxation
#      :lo  |  Lower bound of the interval extension
#      :hi  |  Upper bound of the interval extension
#  :cvgrad  |  Subgradient of the convex relaxation
#  :ccgrad  |  Subgradient of the concave relaxation
#      :MC  |  All of {cv, cc, lo, hi}
#    :grad  |  Both {cvgrad, ccgrad}
#     :all  |  All of {cv, cc, lo, hi, cvgrad, ccgrad} (DEFAULT)

# For example, to get only the convex relaxation and lower bound:
new_func = fgen(expr, [:cv, :lo])
```


### 3) Mutated Inputs

For any call to `fgen`, the keyword argument `mutate` may be used to specify that the generated function
should modify a set of inputs rather than return newly generated vectors (or `CuArray`s). This functionality
is meant to be used if generated functions are incorporated into a larger script and the space for the
outputs has already been preallocated. Mutating functions will require extra inputs at the start of the
input list which correspond to the desired outputs. This works in the following way:

```julia
using SourceCodeMcCormick, Symbolics
Symbolics.@variables x, y
expr = exp(x/y) - (x*y^2)/(y+1)
new_func! = fgen(expr, [:cv, :lo, :cvgrad], mutate=true)

# Inputs to new_func! are as follows:
# {OUT_cv, OUT_lo, OUT_∂x_cv, OUT_∂y_cv, x_cv, x_cc, x_lo, x_hi, y_cv, y_cc, y_lo, y_hi}
```

In this example, `new_func!` will no longer output any information. The results of any calculations are
instead stored in the corresponding input spaces. Note that this only works with vectors and `CuArray`s,
since you cannot preallocate a Float64. Although not necessary for `mutate` to work, this example also
shows that specifying the outputs as `[:cv, :lo, :cvgrad]` means that mutating inputs are only required
for the corresponding set of desired outputs. In thie case, storage for the convex relaxation, lower
bound, subgradient with respect to `x`, and subgradient with respect to `y` must be provided as inputs.
The usual McCormick tuples associated with the variables `x` and `y` are still required, and will always
appear after the mutating inputs. If any constant symbols are used (see `constants` earlier in this
section), the order will be `{{mutating inputs}, {constants}, {McCormick variables}}`, each of which
is sorted alphabetically (and then expanded, for the McCormick variables, as usual).


### 4) Base-Level Variables

In some cases, it may be useful to use `fgen` multiple times to represent a very complicated expression,
with the results connected together---or fed into one another---using generalized McCormick theory. This
approach is not recommended in general, due to the added complexity it entails, but allowances for this
approach are nontheless incorporated into `fgen`. 

Specifically, `fgen` works by compiling a list of variables in the input expression, and then assumes 
that each input variable is a base-level variable (i.e., its relaxation subgradients are 0 in every 
dimension but its own, and 1 for its own dimension). For this reason, subgradient information is not
a required input for generated functions (because it is assumed), but if you use a variable that is
a composite of base-level variables, its subgradients cannot be so simply assumed. Here is an example
of how this functionality may be used (though note that the example expression can easily be handled
with a single call to `fgen`---this is only to show what this functionality does):

```julia
using SourceCodeMcCormick, Symbolics
Symbolics.@variables x, sub

### Full expression: (x+1)^2 * (x-1)

sub_expression = (x+1)^2
new_func1 = fgen(sub_expression) # Assumes that "x" is a base-level variable

#   Inputs to new_func1: 
#   {x_cv, x_cc, x_lo, x_hi}


new_expression = sub * (x-1)
new_func2 = fgen(new_expression, [x]) # Tells fgen that "sub" depends on x

#   Inputs to new_func2: 
#   {sub_cv, sub_cc, sub_lo, sub_hi, ∂sub∂x_cv, ∂sub∂x_cc, x_cv, x_cc, x_lo, x_hi}
```

Particularly, notice that for `new_func2`, it requires the inputs `{sub, x}` split into
their McCormick tuples, but in addition, `fgen` is told that `sub` is not a base-level
variable. I.e., its subgradient cannot be assumed to be 1 in its dimension, as it doesn't
have its own unique dimension. Instead, it is some composite of the true base-level
variable `x`, and thus, the subgradient of `sub`'s convex and concave relaxations in the
`x` dimension must also be provided as inputs.

Relatedly, there may be cases where you would want extra inputs to the generated functions
even if they are not participating in the expression. For example, if you are incorporating
constaints and an objective function into an optimization problem and are using `fgen` to
calculate the relevant relaxations, but the constraints and objective function do not all
use the same set of variables (e.g., some may participate in the objective function but not
a constraint, or vice versa), it may be difficult for the optimizer to know ahead of time
what variables participate in what constraint. One solution to this problem is to give
each generated function full information of all participating variables across all constraints
and the objective function. This is accomplished by specifying base-level variables and
then passing the `all_inputs` keyword argument as `true`. For example:

```julia
using SourceCodeMcCormick, Symbolics
Symbolics.@variables x, y

# Constraint 1: (x + 2)^2 - 3 <= 0
cons1 = (x+2)^2 - 3
f_cons1 = fgen(cons1, [x, y], all_inputs=true)

#   Inputs are:
#   {x_cv, x_cc, x_lo, x_hi, y_cv, y_cc, y_lo, y_hi}


# Constraint 2: (y - 1)/(y + 1) - 0.9 <= 0
cons2 = (y-1)/(y+1) - 0.9
f_cons2 = fgen(cons2, [x, y], all_inputs=true)

#   Inputs are:
#   {x_cv, x_cc, x_lo, x_hi, y_cv, y_cc, y_lo, y_hi}
```

Notice here that although only `x` participated in constraint 1, and only `y` participated in
constraint 2, the input list is the same for both constraints. This is because `[x, y]` was
specified as the base-level variables, and `all_inputs` was set to `true`. By calling `fgen`
in this way, an optimizer that used the generated functions for these constraints would only
need to know that `x` and `y` participated in any/all of the expressions (i.e., the optimizer
could be informed that there are 2 total variables, with no other extra information about the
constraints or objective specifically). The optimizer might then loop over all the constraints
and call the generated functions using the exact same list of inputs, and the generated functions
would simply ignore any of the inputs that are not relevant for the given expression. Adding
these outputs does not impact the speed of the generated functions, since the irrelevant inputs 
aren't used in any calculations.


## The ParBB Algorithm (v0.4)

The intended use of SourceCodeMcCormick is in conjunction with a branch-and-bound algorithm
that is able to make use of relaxations and subgradient information that are calculated in
parallel on a GPU. See the paper reference in the section "Citing SourceCodeMcCormick" for 
a more complete description of the ParBB algorithm. Briefly, ParBB is built as an extension 
of the EAGO solver, and it works by parallelizing the node procesing routines in such a way
that tasks may be performed by a GPU. This section will demonstrate how SourceCodeMcCormick
may be used in conjunction with the ParBB algorithm to accelerate global optimization solutions.

### A) Problem solved using base EAGO

The following example comes from an [EAGO-notebooks page](https://github.com/PSORLab/EAGO-notebooks/blob/master/notebooks/nlpopt_explicit_ann.ipynb)
and involves the optimization of an ANN surrogate model. Here is how the problem is solved
using the base implementation of EAGO, as given in the linked Jupyter Notebook:

```julia
using JuMP, EAGO, GLPK

# Weights associated with the hidden layer
W1 = [ 0.54  -1.97  0.09  -2.14  1.01  -0.58  0.45  0.26;
     -0.81  -0.74  0.63  -1.60 -0.56  -1.05  1.23  0.93;
     -0.11  -0.38 -1.19   0.43  1.21   2.78 -0.06  0.40]

# Weights associated with the output layer
W2 = [-0.91 0.11 0.52]

# Bias associated with the hidden layer
B1 = [-2.698 0.012 2.926]

# Bias associated with the output layer
B2 = -0.46

# Variable bounds (Used to scale variables after optimization)
xLBD = [0.623, 0.093, 0.259, 6.56, 1114,  0.013, 0.127, 0.004]
xUBD = [5.89,  0.5,   1.0,   90,   25000, 0.149, 0.889, 0.049];

# Create the objective function
ann_cpu(p1::T, p2::T, p3::T, p4::T, p5::T, p6::T, p7::T, p8::T) where {T<:Real} = 
    ann_cpu(W1, W2, B1, B2, p1, p2, p3, p4, p5, p6, p7, p8)
function ann_cpu(W1::Matrix{Float64}, W2::Matrix{Float64}, B1::Matrix{Float64}, B2::Float64, 
                p1::T, p2::T, p3::T, p4::T, p5::T, p6::T, p7::T, p8::T) where {T<:Real}
     y1 = W1[1,1]*p1 + W1[1,2]*p2 + W1[1,3]*p3 + W1[1,4]*p4 + W1[1,5]*p5 + W1[1,6]*p6 + W1[1,7]*p7 + W1[1,8]*p8
     y2 = W1[2,1]*p1 + W1[2,2]*p2 + W1[2,3]*p3 + W1[2,4]*p4 + W1[2,5]*p5 + W1[2,6]*p6 + W1[2,7]*p7 + W1[2,8]*p8
     y3 = W1[3,1]*p1 + W1[3,2]*p2 + W1[3,3]*p3 + W1[3,4]*p4 + W1[3,5]*p5 + W1[3,6]*p6 + W1[3,7]*p7 + W1[3,8]*p8

     # Note: the objective is already minimized here, relative to the Jupyter notebook
     return -(B2 + W2[1]*(2/(1+exp(-2*y1+B1[1]))) + W2[2]*(2/(1+exp(-2*y2+B1[2]))) + W2[3]*(2/(1+exp(-2*y3+B1[3]))))
end

# Model construction
factory = () -> EAGO.Optimizer(SubSolvers(r = GLPK.Optimizer()))
model = Model(optimizer_with_attributes(factory, "absolute_tolerance" => 0.001, 
                                                 "output_iterations" => 10000))
register(model,:ann_cpu,8,ann_cpu,autodiff=true)
@variable(model, -1.0 <= p[i=1:8] <= 1.0)
@NLobjective(model, Min, ann_cpu(p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]))

# Solve the model
optimize!(model)
```

Running this code (after an initial compilation run) generates the following output:
```
---------------------------------------------------------------------------------------------------------------------------------
|  Iteration #  |     Nodes     |  Lower Bound  |  Upper Bound  |      Gap      |     Ratio     |     Timer     |   Time Left   |
---------------------------------------------------------------------------------------------------------------------------------
|         10000 |          8059 |    -7.680E-01 |    -7.048E-01 |     6.319E-02 |     8.228E-02 |         13.19 |       3586.81 |
|         20000 |         14199 |    -7.495E-01 |    -7.048E-01 |     4.470E-02 |     5.964E-02 |         17.50 |       3582.50 |
|         30000 |         19415 |    -7.375E-01 |    -7.048E-01 |     3.268E-02 |     4.432E-02 |         20.71 |       3579.29 |
|         40000 |         24013 |    -7.297E-01 |    -7.048E-01 |     2.494E-02 |     3.418E-02 |         23.85 |       3576.15 |
|         50000 |         28091 |    -7.243E-01 |    -7.048E-01 |     1.953E-02 |     2.697E-02 |         26.78 |       3573.22 |
|         60000 |         31383 |    -7.203E-01 |    -7.048E-01 |     1.551E-02 |     2.153E-02 |         29.45 |       3570.55 |
|         70000 |         33939 |    -7.172E-01 |    -7.048E-01 |     1.240E-02 |     1.729E-02 |         32.50 |       3567.50 |
|         80000 |         34973 |    -7.147E-01 |    -7.048E-01 |     9.958E-03 |     1.393E-02 |         35.00 |       3565.00 |
|         90000 |         34771 |    -7.127E-01 |    -7.048E-01 |     7.905E-03 |     1.109E-02 |         37.48 |       3562.52 |
---------------------------------------------------------------------------------------------------------------------------------
|  Iteration #  |     Nodes     |  Lower Bound  |  Upper Bound  |      Gap      |     Ratio     |     Timer     |   Time Left   |
---------------------------------------------------------------------------------------------------------------------------------
|        100000 |         32945 |    -7.110E-01 |    -7.048E-01 |     6.177E-03 |     8.688E-03 |         39.96 |       3560.04 |
|        110000 |         28647 |    -7.095E-01 |    -7.048E-01 |     4.708E-03 |     6.635E-03 |         42.31 |       3557.69 |
|        120000 |         22075 |    -7.082E-01 |    -7.048E-01 |     3.463E-03 |     4.889E-03 |         44.65 |       3555.35 |
|        130000 |         14571 |    -7.072E-01 |    -7.048E-01 |     2.416E-03 |     3.417E-03 |         46.84 |       3553.16 |
|        140000 |          6497 |    -7.063E-01 |    -7.048E-01 |     1.536E-03 |     2.175E-03 |         48.94 |       3551.06 |
|        147217 |             0 |    -7.058E-01 |    -7.048E-01 |     1.000E-03 |     1.417E-03 |         50.38 |       3549.62 |
---------------------------------------------------------------------------------------------------------------------------------

Empty Stack: Exhaustive Search Finished
Optimal Solution Found at Node 2725
Lower Bound: -0.705776904710031
Upper Bound: -0.704776806738834
Solution:
   p[1] = -0.9999999998903418
   p[2] = 0.9999999864966321
   p[3] = -0.9999999997346657
   p[4] = 0.166800526340091
   p[5] = -0.7772932141668214
   p[6] = 0.9999999999319428
   p[7] = 0.9999999998522763
   p[8] = 0.9999999998842807
```

In particular, note that this problem solved in 147,217 iterations after a total time of 50.38
seconds. These times were obtained on a workstation with an Intel i7-9850H processor.

### B) Problem solved with ParBB (subgradient-free)

Next, the same example will be solved by making use of a function generated by 
SourceCodeMcCormick and the subgradient-free routines present in ParBB. This functionality
is described in greater detail in the referenced paper (see "Citing SourceCodeMcCormick").
Briefly, pointwise evaluations of the convex relaxation of the objective function are used
to calculate a valid lower bound in each iteration. By grouping together pointwise evaluations
for a large number of branch-and-bound nodes at the same time and calculating them using
SourceCodeMcCormick generated functions, we can make use of the massive parallelization 
available using GPUs.

```julia
using JuMP, EAGO, SourceCodeMcCormick, Symbolics, DocStringExtensions CUDA

# Import the ParBB algorithm
BASE_FOLDER = dirname(dirname(pathof(SourceCodeMcCormick)))
include(joinpath(BASE_FOLDER, "examples", "ParBB", "extension.jl"))
include(joinpath(BASE_FOLDER, "examples", "ParBB", "subroutines.jl"))
include(joinpath(BASE_FOLDER, "examples", "ParBB", "kernels.jl"))

# Weights associated with the hidden layer
W1 = [ 0.54  -1.97  0.09  -2.14  1.01  -0.58  0.45  0.26;
     -0.81  -0.74  0.63  -1.60 -0.56  -1.05  1.23  0.93;
     -0.11  -0.38 -1.19   0.43  1.21   2.78 -0.06  0.40]

# Weights associated with the output layer
W2 = [-0.91 0.11 0.52]

# Bias associated with the hidden layer
B1 = [-2.698 0.012 2.926]

# Bias associated with the output layer
B2 = -0.46

# Variable bounds (Used to scale variables after optimization)
xLBD = [0.623, 0.093, 0.259, 6.56, 1114,  0.013, 0.127, 0.004]
xUBD = [5.89,  0.5,   1.0,   90,   25000, 0.149, 0.889, 0.049];

# Create a SourceCodeMcCormick generated function for the objective function
Symbolics.@variables x[1:8]
ann_function = fgen(-(B2 + W2[1]*(2/(1+exp(-2*sum(W1[1,i]*x[i] for i=1:8)+B1[1]))) + 
                    W2[2]*(2/(1+exp(-2*sum(W1[2,i]*x[i] for i=1:8)+B1[2]))) + 
                    W2[3]*(2/(1+exp(-2*sum(W1[3,i]*x[i] for i=1:8)+B1[3])))), [:cv])

# Create the objective function
ann_cpu(p1::T, p2::T, p3::T, p4::T, p5::T, p6::T, p7::T, p8::T) where {T<:Real} = 
    ann_cpu(W1, W2, B1, B2, p1, p2, p3, p4, p5, p6, p7, p8)
function ann_cpu(W1::Matrix{Float64}, W2::Matrix{Float64}, B1::Matrix{Float64}, B2::Float64, 
                p1::T, p2::T, p3::T, p4::T, p5::T, p6::T, p7::T, p8::T) where {T<:Real}
     y1 = W1[1,1]*p1 + W1[1,2]*p2 + W1[1,3]*p3 + W1[1,4]*p4 + W1[1,5]*p5 + W1[1,6]*p6 + W1[1,7]*p7 + W1[1,8]*p8
     y2 = W1[2,1]*p1 + W1[2,2]*p2 + W1[2,3]*p3 + W1[2,4]*p4 + W1[2,5]*p5 + W1[2,6]*p6 + W1[2,7]*p7 + W1[2,8]*p8
     y3 = W1[3,1]*p1 + W1[3,2]*p2 + W1[3,3]*p3 + W1[3,4]*p4 + W1[3,5]*p5 + W1[3,6]*p6 + W1[3,7]*p7 + W1[3,8]*p8

     # Note: the objective is already minimized here, relative to the Jupyter notebook
     return -(B2 + W2[1]*(2/(1+exp(-2*y1+B1[1]))) + W2[2]*(2/(1+exp(-2*y2+B1[2]))) + W2[3]*(2/(1+exp(-2*y3+B1[3]))))
end

# Model construction
factory = () -> EAGO.Optimizer(SubSolvers(t = PointwiseGPU(ann_function, # Function returning [:cv] for the objective function
                                                                      8, # Dimensionality of ann_function
                                                        node_limit=8192, # Number of nodes to process per iteration
                                                           gc_freq=50))) # Frequency of garbage collection
model = Model(optimizer_with_attributes(factory, "absolute_tolerance" => 0.001, 
                                                "enable_optimize_hook" => true, # Enables PointwiseGPU extension
                                  "branch_variable" => Bool[true for i in 1:8], # Explicitly tell EAGO to branch on all variables
                                                  "force_global_solve" => true, # Ignore EAGO's problem type detection
                                                    "output_iterations" => 10))
register(model,:ann_cpu,8,ann_cpu,autodiff=true)
@variable(model, -1.0 <= p[i=1:8] <= 1.0)
@NLobjective(model, Min, ann_cpu(p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]))

# Solve the model
optimize!(model)
```

Running this code (after an initial compilation run) generates the following output:
```
---------------------------------------------------------------------------------------------------------------------------------
|  Iteration #  |     Nodes     |  Lower Bound  |  Upper Bound  |      Gap      |     Ratio     |     Timer     |   Time Left   |
---------------------------------------------------------------------------------------------------------------------------------
|            10 |         47818 |    -8.337E-01 |    -7.048E-01 |     1.289E-01 |     1.546E-01 |          0.47 |       3599.53 |
|            20 |        117316 |    -7.703E-01 |    -7.048E-01 |     6.549E-02 |     8.503E-02 |          0.83 |       3599.17 |
|            30 |        162052 |    -7.608E-01 |    -7.048E-01 |     5.604E-02 |     7.365E-02 |          1.26 |       3598.74 |
|            40 |        177018 |    -7.473E-01 |    -7.048E-01 |     4.254E-02 |     5.693E-02 |          1.74 |       3598.26 |
|            50 |        161712 |    -7.411E-01 |    -7.048E-01 |     3.632E-02 |     4.901E-02 |          2.62 |       3597.38 |
|            60 |        128062 |    -7.293E-01 |    -7.048E-01 |     2.453E-02 |     3.363E-02 |          3.17 |       3596.83 |
|            70 |         73936 |    -7.211E-01 |    -7.048E-01 |     1.631E-02 |     2.262E-02 |          3.76 |       3596.24 |
|            80 |         11916 |    -7.180E-01 |    -7.048E-01 |     1.321E-02 |     1.841E-02 |          4.36 |       3595.64 |
|            82 |          4560 |    -7.051E-01 |    -7.048E-01 |     3.648E-04 |     5.173E-04 |          4.42 |       3595.58 |
---------------------------------------------------------------------------------------------------------------------------------

Relative Tolerance Achieved
Optimal Solution Found at Node 1
Lower Bound: -0.7051415840592385
Upper Bound: -0.7047768067379733
Solution:
   p[1] = -0.9999999998903418
   p[2] = 0.9999999864976372
   p[3] = -0.9999999997346648
   p[4] = 0.16680027211461484
   p[5] = -0.777292487172587
   p[6] = 0.9999999999319427
   p[7] = 0.9999999998522755
   p[8] = 0.9999999998842802
```

Using the subgradient-free method, and processing 8192 nodes per iteration, this problem converged in 82
iterations (roughly 671,744 nodes explored), with a total time of 4.42 seconds. These times were obtained
on a workstation with an Intel i7-9850H processor and an NVIDIA Quadro T2000 GPU. Using a GPU with a
greater capacity for double-precision floating point calculations will, of course, improve the overall 
performance of the algorithm, but even with a fairly "typical" GPU such as this, relatively competitive
speed can be obtained. As compared to the base version of EAGO (which makes use of subgradients), this
example ran roughly 11x faster. 

It is also important to note that, because a subgradient-free method was used, the lower bounds for any
individual branch-and-bound node are not as tight as those that can be obtained using subgradient-based
methods. This is one reason why ParBB needed to explore significantly more nodes than the base version
of EAGO (nearly 5x as many nodes were explored in this case). The limitation of not using subgradients
can also cause some optimization problems to converge extremely slowly (see the examples in the paper
referenced in "Citing SourceCodeMcCormick"). For this reason, all standard global optimizers make use
of subgradient information.

### C) Problem solved with ParBB (using subgradients)

NOTE: The use of subgradients in ParBB is an active area of research and remains under development. Associated
papers are forthcoming regarding the incorporation of subgradients into SourceCodeMcCormick and the use
of them within ParBB.

One of the latest developments for SourceCodeMcCormick is the incorporation of subgradient information
into the functions generated by `fgen`. The ability to calculate subgradients using a GPU enables
subgradient-based lower-bounding methods to be used, provided that a routine is available that can
make use of them. Typically, subgradients are used to generate a linear program (LP) that underapproximates
the convex relaxation(s) in a given problem. The LP for a given node is then passed to a dedicated LP
solver such as GLPK as in part (A) of this section.

In ParBB, a large number of branch-and-bound nodes are evaluated simultaneously, meaning subgradients
for many nodes are generated simultaneously, with the results stored in GPU memory. To use these
subgradients, ParBB needs a GPU-based LP solver that is capable of handling many LPs simultaneously: one
for each branch-and-bound node. This has been accomplished by adapting the two-phase simplex method
to perform each individual step on many separate LPs stacked together in the same matrix (similar to
many simplex tableaus stacked on top of each other, or a "stacked tableau"). The individual simplex
steps are performed using custom kernels that parallelize the operations on the GPU. 

Note that the use of the two-phase simplex method is not meant to imply that this method of solving
LPs is superior to the other possible methods. It was implemented primarily as a proof-of-concept
tool to demonstrate how subgradient information could be used within a GPU-accelerated branch-and-bound 
algorithm. Other options are being explored that may be significantly more efficient than the current
implementation. The following shows how SourceCodeMcCormick's new subgradient feature may be used
with ParBB.

```julia
using JuMP, EAGO, SourceCodeMcCormick, Symbolics, DocStringExtensions, CUDA

# Import the ParBB algorithm
BASE_FOLDER = dirname(dirname(pathof(SourceCodeMcCormick)))
include(joinpath(BASE_FOLDER, "examples", "ParBB", "extension.jl"))
include(joinpath(BASE_FOLDER, "examples", "ParBB", "subroutines.jl"))
include(joinpath(BASE_FOLDER, "examples", "ParBB", "kernels.jl"))

# Weights associated with the hidden layer
W1 = [ 0.54  -1.97  0.09  -2.14  1.01  -0.58  0.45  0.26;
     -0.81  -0.74  0.63  -1.60 -0.56  -1.05  1.23  0.93;
     -0.11  -0.38 -1.19   0.43  1.21   2.78 -0.06  0.40]

# Weights associated with the output layer
W2 = [-0.91 0.11 0.52]

# Bias associated with the hidden layer
B1 = [-2.698 0.012 2.926]

# Bias associated with the output layer
B2 = -0.46

# Variable bounds (Used to scale variables after optimization)
xLBD = [0.623, 0.093, 0.259, 6.56, 1114,  0.013, 0.127, 0.004]
xUBD = [5.89,  0.5,   1.0,   90,   25000, 0.149, 0.889, 0.049];

# Create a SourceCodeMcCormick generated function for the objective function
Symbolics.@variables x[1:8]
ann_function! = fgen(-(B2 + W2[1]*(2/(1+exp(-2*sum(W1[1,i]*x[i] for i=1:8)+B1[1]))) + 
                    W2[2]*(2/(1+exp(-2*sum(W1[2,i]*x[i] for i=1:8)+B1[2]))) + 
                    W2[3]*(2/(1+exp(-2*sum(W1[3,i]*x[i] for i=1:8)+B1[3])))), [:cv, :lo, :cvgrad], mutate=true)

# Create the objective function
ann_cpu(p1::T, p2::T, p3::T, p4::T, p5::T, p6::T, p7::T, p8::T) where {T<:Real} = 
    ann_cpu(W1, W2, B1, B2, p1, p2, p3, p4, p5, p6, p7, p8)
function ann_cpu(W1::Matrix{Float64}, W2::Matrix{Float64}, B1::Matrix{Float64}, B2::Float64, 
                p1::T, p2::T, p3::T, p4::T, p5::T, p6::T, p7::T, p8::T) where {T<:Real}
     y1 = W1[1,1]*p1 + W1[1,2]*p2 + W1[1,3]*p3 + W1[1,4]*p4 + W1[1,5]*p5 + W1[1,6]*p6 + W1[1,7]*p7 + W1[1,8]*p8
     y2 = W1[2,1]*p1 + W1[2,2]*p2 + W1[2,3]*p3 + W1[2,4]*p4 + W1[2,5]*p5 + W1[2,6]*p6 + W1[2,7]*p7 + W1[2,8]*p8
     y3 = W1[3,1]*p1 + W1[3,2]*p2 + W1[3,3]*p3 + W1[3,4]*p4 + W1[3,5]*p5 + W1[3,6]*p6 + W1[3,7]*p7 + W1[3,8]*p8

     # Note: the objective is already minimized here, relative to the Jupyter notebook
     return -(B2 + W2[1]*(2/(1+exp(-2*y1+B1[1]))) + W2[2]*(2/(1+exp(-2*y2+B1[2]))) + W2[3]*(2/(1+exp(-2*y3+B1[3]))))
end

# Model construction
factory = () -> EAGO.Optimizer(SubSolvers(; t = SimplexGPU_ObjAndCons(ann_function!, # Mutating function returning [:cv, :lo, :cvgrad] for the objective function
                                                                                  8, # Dimensionality of ann_function!
                                                                    node_limit=8192, # Number of nodes to process per iteration
                                                                       max_cuts=1))) # Number of points/subgradients to evaluate per node
model = Model(optimizer_with_attributes(factory, "enable_optimize_hook" => true,
                                   "branch_variable" => Bool[true for i in 1:8],
                                                   "force_global_solve" => true,
                                                     "output_iterations" => 10))
register(model,:ann_cpu,8,ann_cpu,autodiff=true)
@variable(model, -1.0 <= p[i=1:8] <= 1.0)
@NLobjective(model, Min, ann_cpu(p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]))

# Solve the model
optimize!(model)
```

Running this code (after an initial compilation run) generates the following output:
```
---------------------------------------------------------------------------------------------------------------------------------
|  Iteration #  |     Nodes     |  Lower Bound  |  Upper Bound  |      Gap      |     Ratio     |     Timer     |   Time Left   |
---------------------------------------------------------------------------------------------------------------------------------
|            10 |         16280 |    -7.754E-01 |    -7.048E-01 |     7.067E-02 |     9.113E-02 |          0.45 |       3599.55 |
|            20 |         59342 |    -7.353E-01 |    -7.048E-01 |     3.055E-02 |     4.154E-02 |          1.09 |       3598.91 |
|            30 |         46048 |    -7.108E-01 |    -7.048E-01 |     5.983E-03 |     8.418E-03 |          1.85 |       3598.15 |
|            40 |          2240 |    -7.058E-01 |    -7.048E-01 |     9.852E-04 |     1.396E-03 |          2.33 |       3597.67 |
---------------------------------------------------------------------------------------------------------------------------------

Absolute Tolerance Achieved
Optimal Solution Found at Node 1
Lower Bound: -0.7057619894307733
Upper Bound: -0.7047768067379733
Solution:
   p[1] = -0.9999999998903418
   p[2] = 0.9999999864976372
   p[3] = -0.9999999997346648
   p[4] = 0.16680027211461484
   p[5] = -0.777292487172587
   p[6] = 0.9999999999319427
   p[7] = 0.9999999998522755
   p[8] = 0.9999999998842802
```

As in the previous examples, these results were generated using an Intel i7-9850H processor and an NVIDIA
Quadro T2000 GPU. Effectively, ParBB is running precisely the same lower-bounding routine as the base
version of the global solver EAGO, except that the routine is being performed in parallel on a GPU
rather than serially on the CPU. This example converges in 40 iterations of at most 8192 nodes per iteration,
for a total of roughly 327,680 nodes explored, and reaches its solution in 2.33 seconds. As compared to the
base version of EAGO, this implementation of ParBB solves the problem roughly 22x faster. As in the previous
subsection, it should be noted that a GPU with better double-precision floating-point calculation throughput
would yield even faster results.

It is also interesting to note that roughly twice as many nodes were explored in this example as compared
to the base version of EAGO, despite the same lower-bounding method being used. There are several factors
that could be affecting this result. First, it should be noted that not every iteration of ParBB may be
processing the full 8192 nodes. If fewer than 8192 nodes are available in the branch-and-bound stack,
ParBB will simply use every node in the stack. Thus, the estimate of 8192*40 nodes is conservatively high.
Second, because ParBB processes nodes in parallel, decisions about which node(s) to explore next and
which nodes should be fathomed will almost certainly differ from a serial implementation. Parallel
processing results in processing some nodes that would have been fathomed in a serial implementation,
thereby increasing the total node count for parallel branch-and-bound algorithms. The benefit, then, is
that ParBB is able to make use of the faster computing hardware of GPUs, and even with more nodes being
processed, can converge more quickly overall.


## Limitations (v0.4)

(See "Citing SourceCodeMcCormick" for limitations on a pre-subgradient version of SourceCodeMcCormick.)

- `SCMC` is currently compatible with elementary arithmetic operations +, -, *, /, and the
univariate intrinsic functions ^2 and exp. More diverse functions will be added in the future
- Due to the large number of floating point calculations required to calculate McCormick-based
relaxations, it is highly recommended to use double-precision floating point numbers, including
for operations on a GPU. Since most GPUs are designed for single-precision floating point operation,
forcing double-precision will often result in a significant performance hit. GPUs designed for
scientific computing, with a higher proportion of double-precision-capable cores, are recommended
for optimal performance with `SCMC`.
- Due to the high branching factor of McCormick-based relaxations and the possibility of warp
divergence, there will likely be a slight performance gap between optimization problems with 
variables covering positive-only domains and problems containing variables with mixed domains. 
Additionally, more complicated expressions where the structure of a McCormick relaxation changes 
more frequently with respect to the bounds on its domain may perform worse than expressions 
where the structure of the relaxation is more consistent. 


## Citing SourceCodeMcCormick (v0.3)
Please cite the following paper when using SourceCodeMcCormick.jl. In plain text form this is:
```
Gottlieb, R. X., Xu, P., and Stuber, M. D. Automatic source code generation for deterministic
global optimization with parallel architectures. Optimization Methods and Software, 1–39 (2024).
DOI: 10.1080/10556788.2024.2396297
```
A BibTeX entry is given below:
```bibtex
@Article{doi:10.1080/10556788.2024.2396297,
  author    = {Robert X. Gottlieb, Pengfei Xu, and Matthew D. Stuber},
  journal   = {Optimization Methods and Software},
  title     = {Automatic source code generation for deterministic global optimization with parallel architectures},
  year      = {2024},
  pages     = {1--39},
  doi       = {10.1080/10556788.2024.2396297},
  eprint    = {https://doi.org/10.1080/10556788.2024.2396297},
  publisher = {Taylor \& Francis},
  url       = {https://doi.org/10.1080/10556788.2024.2396297},
}
```


## References

1. M.E. Wilhelm, R.X. Gottlieb, and M.D. Stuber, PSORLab/McCormick.jl (2020), URL
https://github.com/PSORLab/McCormick.jl.
2. T. Besard, C. Foket, and B. De Sutter, Effective extensible programming: Unleashing Julia
on GPUs, IEEE Transactions on Parallel and Distributed Systems (2018).
3. Y. Ma, S. Gowda, R. Anantharaman, C. Laughman, V. Shah, C. Rackauckas, ModelingToolkit: 
A composable graph transformation system for equation-based modeling. arXiv preprint 
arXiv:2103.05244, 2021. doi: 10.48550/ARXIV.2103.05244.