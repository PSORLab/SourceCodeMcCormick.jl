
# This extension is meant to be used with SourceCodeMcCormick to overload
# EAGO's standard branch-and-bound algorithm, to enable the parallel
# evaluation of multiple B&B nodes.
# See also: subroutines.jl, in this folder

using DocStringExtensions, EAGO

abstract type ExtendGPU <: EAGO.ExtensionType end

"""
$(TYPEDEF)

The PointwiseGPU integrator is meant to be paired with the SourceCodeMcCormick
package. A required component of PointwiseGPU is the function `convex_func`, 
which should take arguments corresponding to the McCormick tuple [cc, cv, hi, lo]
for each branch variable in the problem and return a vector of convex
relaxation evaluations of the objective function, of length equal to the
length of the inputs.

$(TYPEDFIELDS)
"""
Base.@kwdef mutable struct PointwiseGPU <: ExtendGPU
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
    "Frequency of garbage collection (number of iterations)"
    gc_freq::Int = 300
    "(In development) Number of points to use for multistarting the NLP solver"
    multistart_points::Int = 1
end

function PointwiseGPU(convex_func, var_count::Int; alpha::Float64 = 0.01, node_limit::Int = 50000, 
                    prepopulate::Bool = true, gc_freq::Int = 300, multistart_points::Int = 1)
    return PointwiseGPU(convex_func, var_count, node_limit, alpha, 
                    Vector{Float64}(undef, node_limit), Vector{Float64}(undef, node_limit), Vector{NodeBB}(undef, node_limit), 0,
                    Matrix{Float64}(undef, node_limit, var_count),
                    Matrix{Float64}(undef, node_limit, var_count), prepopulate, gc_freq, multistart_points)
end


"""
$(TYPEDEF)

The SubgradGPU integrator is meant to be paired with the SourceCodeMcCormick
package. SubgradGPU differs from PointwiseGPU in that SubgradGPU requires
the `convex_func_and_subgrad` term to return both evaluations of the convex
relaxation and evaluations of the subgradient of the convex relaxation.

$(TYPEDFIELDS)
"""
Base.@kwdef mutable struct SubgradGPU <: ExtendGPU
    "A user-defined function taking argument `p` and returning a vector
    of convex evaluations of the objective function [outdated description, [cv, lo, subgrad]]"
    convex_func_and_subgrad
    "Number of decision variables"
    np::Int
    "The number of nodes to evaluate in parallel (default = 10000)"
    node_limit::Int64 = 50000
    "A parameter that changes how far spread out points are. Should be
    in the range (0.0, 1.0]"
    α::Float64 = 0.5
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

function SubgradGPU(convex_func_and_subgrad, var_count::Int; alpha::Float64 = 0.01, node_limit::Int = 50000, 
                    prepopulate::Bool = true, multistart_points::Int = 1)
    return SubgradGPU(convex_func_and_subgrad, var_count, node_limit, alpha, 
                    Vector{Float64}(undef, node_limit), Vector{Float64}(undef, node_limit), Vector{NodeBB}(undef, node_limit), 0,
                    Matrix{Float64}(undef, node_limit, var_count),
                    Matrix{Float64}(undef, node_limit, var_count), prepopulate, multistart_points)
end


"""
$(TYPEDEF)

The SimplexGPU integrator is meant to be paired with the SourceCodeMcCormick
package. SimplexGPU differs from SubgradGPU in that SimplexGPU can handle
inequality constraints, and that relaxations are made tighter by solving
linear programs within the lower bounding routine to make better use of
subgradient information. Like SubgradGPU, SimplexGPU requires the 
`convex_func_and_subgrad` term to return both evaluations of the convex
relaxation and evaluations of the subgradient of the convex relaxation.

$(TYPEDFIELDS)
"""
Base.@kwdef mutable struct SimplexGPU_OnlyObj <: ExtendGPU
    "A user-defined function taking argument `p` and returning a vector
    of convex evaluations of the objective function [outdated description, [cv, lo, subgrad]]"
    convex_func_and_subgrad
    "Number of decision variables"
    np::Int
    "The number of nodes to evaluate in parallel (default = 1024)"
    node_limit::Int64 = 1024
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
    "Flag for stack prepopulation. Good if the total number
    of nodes throughout the solve is expected to be large (default = true)"
    prepopulate::Bool = true
    "Total number of cuts to do on each node"
    max_cuts::Int = 3
    "Frequency of garbage collection (number of iterations)"
    gc_freq::Int = 15
    "(In development) Number of points to use for multistarting the NLP solver"
    multistart_points::Int = 1
    relax_time::Float64 = 0.0
    opt_time::Float64 = 0.0
    lower_counter::Int = 0
    node_counter::Int = 0
end

function SimplexGPU_OnlyObj(convex_func_and_subgrad, var_count::Int; node_limit::Int = 1024, 
                    prepopulate::Bool = true, max_cuts::Int = 3, gc_freq::Int = 15, 
                    multistart_points::Int = 1)
    return SimplexGPU_OnlyObj(convex_func_and_subgrad, var_count, node_limit, 
                    Vector{Float64}(undef, node_limit), Vector{Float64}(undef, node_limit), Vector{NodeBB}(undef, node_limit), 0,
                    Matrix{Float64}(undef, node_limit, var_count),
                    Matrix{Float64}(undef, node_limit, var_count), prepopulate, max_cuts, gc_freq, multistart_points, 0.0, 0.0, 0, 0)
end

"""
$(TYPEDEF)

The SimplexGPU_ObjAndCons structure is meant to handle optimization problems
with nontrivial constraints as well as a potentially nonlinear objective
function. Note that this struct requires the functions representing the
objective function and constraints to mutate arguments, rather than return
a tuple of results. SimplexGPU_ObjAndCons is not designed to handle mixed-integer
problems; NLPs only.

$(TYPEDFIELDS)
"""
Base.@kwdef mutable struct SimplexGPU_ObjAndCons <: ExtendGPU
    "A SCMC-generated or user-defined function taking arguments [cv, lo, [cv_subgrad]..., p...],
    which modifies `cv` to hold the convex relaxation of the objective function, `lo` to hold 
    the lower bound of the inclusion monotonic interval extension of the objective function, 
    and n instances of `cv_subgrad` that will hold the n subgradients of the convex relaxation 
    of the objective function (where n is the dimensionality of the problem), all evaluated at
    points `p`"
    obj_fun
    "A vector of SCMC-generated or user-defined functions, each with the same form as `obj_fun`,
    but with arguments [cv, [cv_subgrad]..., p...], representing all of the LEQ inequality constraints"
    leq_cons
    "A vector of SCMC-generated or user-defined functions, taking arguments [cc, [cc_subgrad]..., p...],
    defined similarly to the objective function and GEQ constraints, representing all of the 
    GEQ inequality constraints"
    geq_cons
    "A vector of SCMC-generated or user-defined functions, taking arguments 
    [cv, cc, [cv_subgrad]..., [cc_subgrad]..., p...], with terms defined similarly to 
    the objective function and inequality constraints, representing all of the equality constraints"
    eq_cons
    "Number of decision variables"
    np::Int
    "The number of nodes to evaluate in parallel (default = 1024)"
    node_limit::Int64 = 1024
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
    "Flag for stack prepopulation. Good if the total number
    of nodes throughout the solve is expected to be large (default = true)"
    prepopulate::Bool = true
    "Total number of cuts to do on each node"
    max_cuts::Int = 3
    "Frequency of garbage collection (number of iterations)"
    gc_freq::Int = 15
    "(In development) Number of points to use for multistarting the NLP solver"
    multistart_points::Int = 1
    relax_time::Float64 = 0.0
    opt_time::Float64 = 0.0
    lower_counter::Int = 0
    node_counter::Int = 0
end

function SimplexGPU_ObjAndCons(obj_fun, var_count::Int; geq_cons=[], leq_cons=[], eq_cons=[], node_limit::Int = 1024,
                        prepopulate::Bool = true, max_cuts::Int = 3, gc_freq::Int = 15, multistart_points::Int = 1)
    return SimplexGPU_ObjAndCons(obj_fun, leq_cons, geq_cons, eq_cons, var_count, node_limit, 
                    Vector{Float64}(undef, node_limit), Vector{Float64}(undef, node_limit), Vector{NodeBB}(undef, node_limit), 0,
                    Matrix{Float64}(undef, node_limit, var_count),
                    Matrix{Float64}(undef, node_limit, var_count), prepopulate, max_cuts, gc_freq, multistart_points, 0.0, 0.0, 0, 0)
end






Base.@kwdef mutable struct SimplexGPU_ObjOnly_Mat <: ExtendGPU
    "A SCMC-generated or user-defined function taking arguments [cv, lo, [cv_subgrad]..., p...],
    which modifies `cv` to hold the convex relaxation of the objective function, `lo` to hold 
    the lower bound of the inclusion monotonic interval extension of the objective function, 
    and n instances of `cv_subgrad` that will hold the n subgradients of the convex relaxation 
    of the objective function (where n is the dimensionality of the problem), all evaluated at
    points `p`"
    obj_fun
    "Number of decision variables"
    np::Int
    "The number of nodes to evaluate in parallel (default = 1024)"
    node_limit::Int64 = 1024
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
    "Flag for stack prepopulation. Good if the total number
    of nodes throughout the solve is expected to be large (default = true)"
    prepopulate::Bool = true
    "Total number of cuts to do on each node"
    max_cuts::Int = 3
    "Frequency of garbage collection (number of iterations)"
    gc_freq::Int = 15
    "(In development) Number of points to use for multistarting the NLP solver"
    multistart_points::Int = 1
    relax_time::Float64 = 0.0
    opt_time::Float64 = 0.0
    lower_counter::Int = 0
    node_counter::Int = 0
end

function SimplexGPU_ObjOnly_Mat(obj_fun, var_count::Int; node_limit::Int = 1024,
                        prepopulate::Bool = true, max_cuts::Int = 3, gc_freq::Int = 15, multistart_points::Int = 1)
    return SimplexGPU_ObjOnly_Mat(obj_fun, var_count, node_limit, 
                    Vector{Float64}(undef, node_limit), Vector{Float64}(undef, node_limit), Vector{NodeBB}(undef, node_limit), 0,
                    Matrix{Float64}(undef, node_limit, var_count),
                    Matrix{Float64}(undef, node_limit, var_count), prepopulate, max_cuts, gc_freq, multistart_points, 0.0, 0.0, 0 ,0)
end


"""
$(TYPEDEF)

This is a testing method/struct, to see if we can check fewer points per node
when we construct the LPs and still get all the same benefits. The normal
SimplexGPU method uses 2n+1 points, where n is the problem dimensionality.
This method only uses a single point in the center of the node, and can
therefore get away with more simultaneous LPs, since each one is significantly
smaller.

$(TYPEDFIELDS)
"""
Base.@kwdef mutable struct SimplexGPU_Single <: ExtendGPU
    "A user-defined function taking argument `p` and returning a vector
    of convex evaluations of the objective function [outdated description, [cv, lo, subgrad]]"
    convex_func_and_subgrad
    "Number of decision variables"
    np::Int
    "The number of nodes to evaluate in parallel (default = 2500)"
    node_limit::Int64 = 2500
    "A parameter that changes how far spread out points are. Should be
    in the range (0.0, 1.0]"
    α::Float64 = 0.5
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
    "Flag for stack prepopulation. Good if the total number
    of nodes throughout the solve is expected to be large (default = true)"
    prepopulate::Bool = true
    "Total number of cuts to do on each node"
    max_cuts::Int = 3
    "(In development) Number of points to use for multistarting the NLP solver"
    multistart_points::Int = 1
end

function SimplexGPU_Single(convex_func_and_subgrad, var_count::Int; alpha::Float64 = 0.01, node_limit::Int = 2500, 
                    prepopulate::Bool = true, max_cuts::Int = 3, multistart_points::Int = 1)
    return SimplexGPU_Single(convex_func_and_subgrad, var_count, node_limit, alpha, 
                    Vector{Float64}(undef, node_limit), Vector{Float64}(undef, node_limit), Vector{NodeBB}(undef, node_limit), 0,
                    Matrix{Float64}(undef, node_limit, var_count),
                    Matrix{Float64}(undef, node_limit, var_count), prepopulate, max_cuts, multistart_points)
end