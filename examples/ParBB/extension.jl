
# This extension is meant to be used with SourceCodeMcCormick to overload
# EAGO's standard branch-and-bound algorithm, to enable the parallel
# evaluation of multiple B&B nodes.
# See also: subroutines.jl, in this folder

using DocStringExtensions, EAGO
"""
$(TYPEDEF)

The ExtendGPU integrator is meant to be paired with the SourceCodeMcCormick
package. A required component of ExtendGPU is the function `convex_func`, 
which should take arguments corresponding to the McCormick tuple [cc, cv, hi, lo]
for each branch variable in the problem and return a vector of convex
relaxation evaluations of the objective function, of length equal to the
length of the inputs.

$(TYPEDFIELDS)
"""
Base.@kwdef mutable struct ExtendGPU <: EAGO.ExtensionType
    "A user-defined function taking argument `p` and returning a vector
    of convex evaluations of the objective function"
    convex_func
    "Number of decision variables"
    np::Int
    "The number of nodes to evaluate in parallel (default = 10000)"
    node_limit::Int64 = 50000
    "A parameter from Song et al. 2021 that determines how spread out 
    blackbox points are (default = 0.01)"
    α::Float64 = 0.01
    "Lower bound storage to hold calculated lower bounds for multiple nodes."
    lower_bound_storage::Vector{Float64} = Vector{Float64}()
    "Upper bound storage to hold calculated upper bounds for multiple nodes."
    upper_bound_storage::Vector{Float64} = Vector{Float64}()
    "Node storage to hold individual nodes outside of the main stack"
    node_storage::Vector{NodeBB} = Vector{NodeBB}()
    "An internal tracker of nodes in internal storage"
    node_len::Int = 0
    "Variable lower bounds to evaluate"
    all_lvbs::Matrix{Float64} = Matrix{Float64}()
    "Variable upper bounds to evaluate"
    all_uvbs::Matrix{Float64} = Matrix{Float64}()
    "Internal tracker for the count in the main stack"
    # node_count::Int = 0
    "Flag for stack prepopulation. Good if the total number
    of nodes throughout the solve is expected to be large (default = true)"
    prepopulate::Bool = true
    "(In development) Number of points to use for multistarting the NLP solver"
    multistart_points::Int = 1
end

function ExtendGPU(convex_func, var_count::Int; alpha::Float64 = 0.01, node_limit::Int = 50000, 
                    prepopulate::Bool = true, multistart_points::Int = 1)
    return ExtendGPU(convex_func, var_count, node_limit, alpha, 
                    Vector{Float64}(undef, node_limit), Vector{Float64}(undef, node_limit), Vector{NodeBB}(undef, node_limit), 0,
                    Matrix{Float64}(undef, node_limit, var_count),
                    Matrix{Float64}(undef, node_limit, var_count), prepopulate, multistart_points)
end