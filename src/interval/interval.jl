#=
Rules for constructing interval bounding expressions
=#

# Structure used to indicate an overload with intervals is preferable
struct IntervalTransform <: AbstractTransform end

# Creates names for interval state variables
function var_names(::IntervalTransform, s::String)
    sL = Symbol(s*"_lo")
    sU = Symbol(s*"_hi")
    sL, sU
end

function var_names(::IntervalTransform, s::Number)
    sL = s
    sU = s
    sL, sU
end

include(joinpath(@__DIR__, "rules.jl"))