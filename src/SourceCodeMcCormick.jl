
module SourceCodeMcCormick

using ModelingToolkit
using SymbolicUtils.Code
# TODO: Need to import Assignment and other stuff probably
# Check out the functionality in ModelingToolkit.jl and Symbolics.jl

abstract type AbstractTransform end

# ADD documentation for generic function here
function transform_rule end


export McCormickIntervalTransform

export apply_transform, extract_terms, genvar, genparam, get_name

include(joinpath(@__DIR__, "interval", "interval.jl"))
include(joinpath(@__DIR__, "relaxation", "relaxation.jl"))
include(joinpath(@__DIR__, "transform", "transform.jl"))

end