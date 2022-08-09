
module SourceCodeMcCormick

using ModelingToolkit
using SymbolicUtils.Code
using IfElse
using DocStringExtensions

# TODO: Need to import Assignment and other stuff probably
# Check out the functionality in ModelingToolkit.jl and Symbolics.jl

abstract type AbstractTransform end

# ADD documentation for generic function here
function transform_rule end


export McCormickIntervalTransform

export apply_transform, extract_terms, genvar, genparam, get_name,
        factor!, binarize!, pull_vars, shrink_eqs, convex_evaluator,
        all_evaluators

include(joinpath(@__DIR__, "interval", "interval.jl"))
include(joinpath(@__DIR__, "relaxation", "relaxation.jl"))
include(joinpath(@__DIR__, "transform", "transform.jl"))

end