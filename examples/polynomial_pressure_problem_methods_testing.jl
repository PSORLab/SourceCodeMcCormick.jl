
#############################################################################
# This code uses the GPU to speed up the solve time for the polynomial 
# pressure optimization example described in Alvarez2011.
#############################################################################

# Import the necessary packages
using JuMP, EAGO
using Symbolics, SourceCodeMcCormick, CUDA, CSV, DataFrames

# Import the ParBB algorithm
include(joinpath(@__DIR__, "ParBB", "extension.jl"))
include(joinpath(@__DIR__, "ParBB", "subroutines.jl"))

# Import the pressure data for the polynomial equation
pressure_data = CSV.read(joinpath(@__DIR__, "polynomial_pressure_data.csv"), DataFrame)

# Adjust CUDA settings to make sure no scalar calculations occur
CUDA.allowscalar(false)

# Set up symbolic variables for the problem
Symbolics.@variables var_a0, var_a1, var_a2, var_b0, var_b1, var_b2, awval, aTval
func = fgen(exp(var_a0 + var_a1*awval + var_a2*awval^2 + (1/aTval)*(var_b0 + var_b1*awval + var_b2*awval^2)), constants=[awval, aTval])

Symbolics.@variables calc, adata
SSE_func = fgen(((calc - adata)/adata)^2, [var_a0, var_a1, var_a2, var_b0, var_b1, var_b2], [:cv, :lo, :cvgrad], constants=[adata])

# Set up simpler versions for the pointwise GPU method
func_simple = fgen(exp(var_a0 + var_a1*awval + var_a2*awval^2 + (1/aTval)*(var_b0 + var_b1*awval + var_b2*awval^2)), [:MC], constants=[awval, aTval])
SSE_func_simple = fgen(((calc - adata)/adata)^2, [var_a0, var_a1, var_a2, var_b0, var_b1, var_b2], [:cv], constants=[adata])

@variables a0, a1, a2, b0, b1, b2, data, W, T
expr = exp(a0 + a1*W + a2*W^2 + (1/T)*(b0 + b1*W + b2*W^2))

new_func = fgen(((expr-data)/data)^2, constants=[data, W, T])

# Set up the GPU-compatible L2norm calculation
L2norm(p...) = L2norm(pressure_data, p...)
function L2norm(pressure_data::DataFrame, p...)
    pset = pressure_data.pressure
    wset = pressure_data.w
    Tset = pressure_data.T

    SSE_cv = CUDA.zeros(Float64, length(p[1]))
    for i = 1:length(pset)
        SSE_cv .+= SSE_func_simple(pset[i], func_simple(Tset[i], wset[i], p...)...)
    end
    return SSE_cv
end

# Set up the more complicated version that includes subgradients
L2norm_grad(p...) = L2norm_grad(pressure_data, p...)
function L2norm_grad(pressure_data::DataFrame, p...)
    pset = pressure_data.pressure
    wset = pressure_data.w
    Tset = pressure_data.T

    diff_SSE_cv = similar(p[1])
    diff_SSE_lo = similar(p[1])
    diff_SSE_cvgrad_a0 = similar(p[1])
    diff_SSE_cvgrad_a1 = similar(p[1])
    diff_SSE_cvgrad_a2 = similar(p[1])
    diff_SSE_cvgrad_b0 = similar(p[1])
    diff_SSE_cvgrad_b1 = similar(p[1])
    diff_SSE_cvgrad_b2 = similar(p[1])

    SSE_cv = CUDA.zeros(Float64, length(p[1]))
    SSE_lo = CUDA.zeros(Float64, length(p[1]))
    SSE_cvgrad_a0 = CUDA.zeros(Float64, length(p[1]))
    SSE_cvgrad_a1 = CUDA.zeros(Float64, length(p[1]))
    SSE_cvgrad_a2 = CUDA.zeros(Float64, length(p[1]))
    SSE_cvgrad_b0 = CUDA.zeros(Float64, length(p[1]))
    SSE_cvgrad_b1 = CUDA.zeros(Float64, length(p[1]))
    SSE_cvgrad_b2 = CUDA.zeros(Float64, length(p[1]))


    for i = 1:length(pset)
        diff_SSE_cv, diff_SSE_lo, diff_SSE_cvgrad_a0, diff_SSE_cvgrad_a1, 
        diff_SSE_cvgrad_a2, diff_SSE_cvgrad_b0, diff_SSE_cvgrad_b1, 
        diff_SSE_cvgrad_b2 = SSE_func(pset[i], func(Tset[i], wset[i], p...)...)

        SSE_cv .+= diff_SSE_cv
        SSE_lo .+= diff_SSE_lo
        SSE_cvgrad_a0 .+= diff_SSE_cvgrad_a0
        SSE_cvgrad_a1 .+= diff_SSE_cvgrad_a1
        SSE_cvgrad_a2 .+= diff_SSE_cvgrad_a2
        SSE_cvgrad_b0 .+= diff_SSE_cvgrad_b0
        SSE_cvgrad_b1 .+= diff_SSE_cvgrad_b1
        SSE_cvgrad_b2 .+= diff_SSE_cvgrad_b2
    end
    return SSE_cv, SSE_lo, SSE_cvgrad_a0, SSE_cvgrad_a1, SSE_cvgrad_a2, SSE_cvgrad_b0, SSE_cvgrad_b1, SSE_cvgrad_b2
end

# CPU version of the function, to be used for IPOPT
L2norm_cpu(p1::T, p2::T, p3::T, p4::T, p5::T, p6::T) where {T<:Real} = L2norm_cpu(pressure_data, p1, p2, p3, p4, p5, p6)
function L2norm_cpu(pressure_data::DataFrame, p1::T, p2::T, p3::T, p4::T, p5::T, p6::T) where {T<:Real}
    SSE = zero(T)
    pset = pressure_data.pressure
    wset = pressure_data.w
    Tset = pressure_data.T

    for i = 1:length(pset)
        temp = exp(p1 + p2*wset[i] + p3*wset[i]^2 + (1/Tset[i])*(p4 + p5*wset[i] + p6*wset[i]^2))

        SSE += ((temp - pset[i])/pset[i])^2
    end
    return SSE
end


# Run the problem once with a short time_limit to compile all necessary functions
factory = () -> EAGO.Optimizer(SubSolvers(; t = PointwiseGPU(L2norm, 6)))
opt = optimizer_with_attributes(factory, "enable_optimize_hook" => true,
                                         "branch_variable" => Bool[true for i in 1:6],
                                         "force_global_solve" => true,
                                         "node_limit" => Int(3e8),
                                         "time_limit" => 10.0,
                                         "output_iterations" => 10000)
model = Model(opt)
register(model,:L2norm_cpu,6,L2norm_cpu,autodiff=true)

# Set bounds to be +/- 0.1 in every dimension, from reported results
reported = [8.7369, 27.0375, -21.4172, -2432.1378, -6955.3785, 4525.9568]
lb = reported .- 0.1
ub = reported .+ 0.1
@variable(model, lb[i]<= x[i=1:6] <= ub[i] )
@NLobjective(model, Min, L2norm_cpu(x[1], x[2], x[3], x[4], x[5], x[6]))
optimize!(model)

# Real run to get times without compilation
factory = () -> EAGO.Optimizer(SubSolvers(; t = PointwiseGPU(L2norm, 6)))
opt = optimizer_with_attributes(factory, "enable_optimize_hook" => true,
                                         "branch_variable" => Bool[true for i in 1:6],
                                         "force_global_solve" => true,
                                         "node_limit" => Int(3e8),
                                         "time_limit" => 60.0,
                                         "output_iterations" => 10,
                                         "log_on" => true)
model = Model(opt)
register(model,:L2norm_cpu,6,L2norm_cpu,autodiff=true)

# Set bounds to be +/- 0.1 in every dimension, from reported results
reported = [8.7369, 27.0375, -21.4172, -2432.1378, -6955.3785, 4525.9568]
lb = reported .- 0.1
ub = reported .+ 0.1
@variable(model, lb[i]<= x[i=1:6] <= ub[i] )
@NLobjective(model, Min, L2norm_cpu(x[1], x[2], x[3], x[4], x[5], x[6]))
optimize!(model)


#########################################################################################
factory = () -> EAGO.Optimizer(SubSolvers(; t = SubgradGPU(L2norm_grad, 6)))
opt = optimizer_with_attributes(factory, "enable_optimize_hook" => true,
                                         "branch_variable" => Bool[true for i in 1:6],
                                         "force_global_solve" => true,
                                         "node_limit" => Int(3e8),
                                         "time_limit" => 10.0,
                                         "output_iterations" => 10000)
model = Model(opt)
register(model,:L2norm_cpu,6,L2norm_cpu,autodiff=true)

# Set bounds to be +/- 0.1 in every dimension, from reported results
reported = [8.7369, 27.0375, -21.4172, -2432.1378, -6955.3785, 4525.9568]
lb = reported .- 0.1
ub = reported .+ 0.1
@variable(model, lb[i]<= x[i=1:6] <= ub[i] )
@NLobjective(model, Min, L2norm_cpu(x[1], x[2], x[3], x[4], x[5], x[6]))
optimize!(model)

# Real run to get times without compilation
factory = () -> EAGO.Optimizer(SubSolvers(; t = SubgradGPU(L2norm_grad, 6, node_limit=1000)))
opt = optimizer_with_attributes(factory, "enable_optimize_hook" => true,
                                         "branch_variable" => Bool[true for i in 1:6],
                                         "force_global_solve" => true,
                                         "node_limit" => Int(3e8),
                                         "time_limit" => 60.0,
                                         "output_iterations" => 10,
                                         "log_on" => true)
model = Model(opt)
register(model,:L2norm_cpu,6,L2norm_cpu,autodiff=true)

# Set bounds to be +/- 0.1 in every dimension, from reported results
reported = [8.7369, 27.0375, -21.4172, -2432.1378, -6955.3785, 4525.9568]
lb = reported .- 0.1
ub = reported .+ 0.1
@variable(model, lb[i]<= x[i=1:6] <= ub[i] )
@NLobjective(model, Min, L2norm_cpu(x[1], x[2], x[3], x[4], x[5], x[6]))
optimize!(model)

#########################################################################################
factory = () -> EAGO.Optimizer(SubSolvers(; t = SimplexGPU(L2norm_grad, 6)))
opt = optimizer_with_attributes(factory, "enable_optimize_hook" => true,
                                         "branch_variable" => Bool[true for i in 1:6],
                                         "force_global_solve" => true,
                                         "node_limit" => Int(3e8),
                                         "time_limit" => 10.0,
                                         "output_iterations" => 10000)
model = Model(opt)
register(model,:L2norm_cpu,6,L2norm_cpu,autodiff=true)

# Set bounds to be +/- 0.1 in every dimension, from reported results
reported = [8.7369, 27.0375, -21.4172, -2432.1378, -6955.3785, 4525.9568]
lb = reported .- 0.1
ub = reported .+ 0.1
@variable(model, lb[i]<= x[i=1:6] <= ub[i] )
@NLobjective(model, Min, L2norm_cpu(x[1], x[2], x[3], x[4], x[5], x[6]))
optimize!(model)

# Real run to get times without compilation
factory = () -> EAGO.Optimizer(SubSolvers(; t = SimplexGPU(L2norm_grad, 6, node_limit=1000)))
opt = optimizer_with_attributes(factory, "enable_optimize_hook" => true,
                                         "branch_variable" => Bool[true for i in 1:6],
                                         "force_global_solve" => true,
                                         "node_limit" => Int(3e8),
                                         "time_limit" => 60.0,
                                         "output_iterations" => 10,
                                         "log_on" => true)
model = Model(opt)
register(model,:L2norm_cpu,6,L2norm_cpu,autodiff=true)

# Set bounds to be +/- 0.1 in every dimension, from reported results
reported = [8.7369, 27.0375, -21.4172, -2432.1378, -6955.3785, 4525.9568]
lb = reported .- 0.1
ub = reported .+ 0.1
@variable(model, lb[i]<= x[i=1:6] <= ub[i] )
@NLobjective(model, Min, L2norm_cpu(x[1], x[2], x[3], x[4], x[5], x[6]))
optimize!(model)

#########################################################################################
factory = () -> EAGO.Optimizer(SubSolvers(; t = SimplexGPU_Single(L2norm_grad, 6, node_limit=2500)))
opt = optimizer_with_attributes(factory, "enable_optimize_hook" => true,
                                         "branch_variable" => Bool[true for i in 1:6],
                                         "force_global_solve" => true,
                                         "node_limit" => Int(3e8),
                                         "time_limit" => 10.0,
                                         "output_iterations" => 10000)
model = Model(opt)
register(model,:L2norm_cpu,6,L2norm_cpu,autodiff=true)

# Set bounds to be +/- 0.1 in every dimension, from reported results
reported = [8.7369, 27.0375, -21.4172, -2432.1378, -6955.3785, 4525.9568]
lb = reported .- 0.1
ub = reported .+ 0.1
@variable(model, lb[i]<= x[i=1:6] <= ub[i] )
@NLobjective(model, Min, L2norm_cpu(x[1], x[2], x[3], x[4], x[5], x[6]))
optimize!(model)

# Real run to get times without compilation
factory = () -> EAGO.Optimizer(SubSolvers(; t = SimplexGPU_Single(L2norm_grad, 6, node_limit=2500)))
opt = optimizer_with_attributes(factory, "enable_optimize_hook" => true,
                                         "branch_variable" => Bool[true for i in 1:6],
                                         "force_global_solve" => true,
                                         "node_limit" => Int(3e8),
                                         "time_limit" => 60.0,
                                         "output_iterations" => 10,
                                         "log_on" => true)
model = Model(opt)
register(model,:L2norm_cpu,6,L2norm_cpu,autodiff=true)

# Set bounds to be +/- 0.1 in every dimension, from reported results
reported = [8.7369, 27.0375, -21.4172, -2432.1378, -6955.3785, 4525.9568]
lb = reported .- 0.1
ub = reported .+ 0.1
@variable(model, lb[i]<= x[i=1:6] <= ub[i] )
@NLobjective(model, Min, L2norm_cpu(x[1], x[2], x[3], x[4], x[5], x[6]))
optimize!(model)

# L2norm_grad_test(8.7369, 8.7369, 8.6369, 8.8369, 
#             27.0375, 27.0375, 26.9375, 27.137500000000003, 
#             -21.4172, -21.4172, -21.517200000000003, -21.3172, 
#             -2432.1378, -2432.1378, -2432.2378, -2432.0378, 
#             -6955.3785, -6955.3785, -6955.4785, -6955.278499999999, 
#             4525.9568, 4525.9568, 4525.8568, 4526.0568)

# L2norm_grad_test(8.7369, 8.7369, 8.7369, 8.7369,
#             27.0375, 27.0375, 27.0375, 27.0375,
#             -21.4172, -21.4172, -21.4172, -21.4172, 
#             -2432.1378, -2432.1378, -2432.1378, -2432.1378, 
#             -6955.3785, -6955.3785, -6955.3785, -6955.3785, 
#             4525.9568, 4525.9568, 4525.9568, 4525.9568)

# L2norm_grad_test(p...) = L2norm_grad_test(pressure_data, p...)
# function L2norm_grad_test(pressure_data::DataFrame, p...)
#     pset = pressure_data.pressure
#     wset = pressure_data.w
#     Tset = pressure_data.T

#     # diff_SSE_cv = similar(p[1])
#     # diff_SSE_lo = similar(p[1])
#     # diff_SSE_cvgrad_a0 = similar(p[1])
#     # diff_SSE_cvgrad_a1 = similar(p[1])
#     # diff_SSE_cvgrad_a2 = similar(p[1])
#     # diff_SSE_cvgrad_b0 = similar(p[1])
#     # diff_SSE_cvgrad_b1 = similar(p[1])
#     # diff_SSE_cvgrad_b2 = similar(p[1])

#     SSE_cv = CUDA.zeros(Float64, length(p[1]))
#     SSE_lo = CUDA.zeros(Float64, length(p[1]))
#     SSE_cvgrad_a0 = CUDA.zeros(Float64, length(p[1]))
#     SSE_cvgrad_a1 = CUDA.zeros(Float64, length(p[1]))
#     SSE_cvgrad_a2 = CUDA.zeros(Float64, length(p[1]))
#     SSE_cvgrad_b0 = CUDA.zeros(Float64, length(p[1]))
#     SSE_cvgrad_b1 = CUDA.zeros(Float64, length(p[1]))
#     SSE_cvgrad_b2 = CUDA.zeros(Float64, length(p[1]))


#     for i = 1:length(pset)
#         diff_SSE_cv, diff_SSE_lo, diff_SSE_cvgrad_a0, diff_SSE_cvgrad_a1, 
#         diff_SSE_cvgrad_a2, diff_SSE_cvgrad_b0, diff_SSE_cvgrad_b1, 
#         diff_SSE_cvgrad_b2 = SSE_func(pset[i], func(Tset[i], wset[i], p...)...)

#         # func_out = func(Tset[i], wset[i], p...)
#         # @show func_out
#         # @show pset[i]
#         # @show SSE_func(pset[i], func_out...)
#         # @show nothing
#         # error()

#         # @show diff_SSE_lo

#         SSE_cv .+= diff_SSE_cv
#         SSE_lo .+= diff_SSE_lo
#         SSE_cvgrad_a0 .+= diff_SSE_cvgrad_a0
#         SSE_cvgrad_a1 .+= diff_SSE_cvgrad_a1
#         SSE_cvgrad_a2 .+= diff_SSE_cvgrad_a2
#         SSE_cvgrad_b0 .+= diff_SSE_cvgrad_b0
#         SSE_cvgrad_b1 .+= diff_SSE_cvgrad_b1
#         SSE_cvgrad_b2 .+= diff_SSE_cvgrad_b2
#     end
#     return SSE_cv, SSE_lo, SSE_cvgrad_a0, SSE_cvgrad_a1, SSE_cvgrad_a2, SSE_cvgrad_b0, SSE_cvgrad_b1, SSE_cvgrad_b2
# end