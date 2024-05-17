
module SourceCodeMcCormick

# using ModelingToolkit
import ModelingToolkit: Equation, ODESystem, @named, toparam, iv_from_nested_derivative, value, collect_vars!
using Symbolics
using SymbolicUtils.Code
using IfElse
using DocStringExtensions
using Graphs
using CUDA
import Dates

import SymbolicUtils: BasicSymbolic, exprtype, SYM, TERM, ADD, MUL, POW, DIV

import SymbolicUtils: BasicSymbolic, exprtype, SYM, TERM, ADD, MUL, POW, DIV

"""
    AbstractTransform

An abstract type holding possible transform options. Current options are:
- `IntervalTransform`: Only lower/upper bounds, results in 2x as many expressions/equations 
        (in simple cases)
- `McCormickIntervalTransform`: Lower/upper bounds and convex/concave relaxations, results in
        4x as many expressions/equations (in simple cases)
"""
abstract type AbstractTransform end

"""
    transform_rule(::AbstractTransform, ::Equation)
    transform_rule(::AbstractTransform, ::Vector{Equation})
    transform_rule(::AbstractTransform, ::ModelingToolkit.ODESystem)
    transform_rule(::AbstractTransform, ::Num)

Apply the desired transformation to a given expression, equation, set of equations, or ODESystem.
If the `AbstractTransform` is an `IntervalTransform`, all symbols are split into lower and upper
bounds, and the number of expressions/equations is initially doubled. `SourceCodeMcCormick` uses
the auxiliary variable method, so factorable expressions may be separated into multiple expressions
through auxiliary variables. If a `McCormickIntervalTransform` is used, all symbols are split into
lower and upper bounds and convex and concave relaxations. The number of expressions/equations
will thus be initially multiplied by 4, though additional expressions with auxiliary variables will
appear if the expression's complexity warrants it.

In either case, `shrink_eqs` can be used to progressively substitute expressions into one another
to eliminate auxiliary variables and shrink the total number of equations.
"""
function transform_rule end


include(joinpath(@__DIR__, "interval", "interval.jl"))
include(joinpath(@__DIR__, "relaxation", "relaxation.jl"))
include(joinpath(@__DIR__, "transform", "transform.jl"))

export McCormickIntervalTransform, IntervalTransform

export apply_transform, all_evaluators, convex_evaluator, extract_terms, 
        genvar, genparam, get_name, factor, binarize!, pull_vars, shrink_eqs
        
end