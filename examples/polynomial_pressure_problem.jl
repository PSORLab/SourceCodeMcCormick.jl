
#############################################################################
# This code uses the GPU to speed up the solve time for the polynomial 
# pressure optimization example described in Alvarez2011.
#############################################################################

# Import the necessary packages
using JuMP, EAGO
using Symbolics, SourceCodeMcCormick, CUDA

# Import the ParBB algorithm
include(joinpath(@__DIR__, "ParBB", "extension.jl"))
include(joinpath(@__DIR__, "ParBB", "subroutines.jl"))

# Import the pressure data for the polynomial equation
pressure_data = CSV.read(joinpath(@__DIR__, "polynomial_pressure_data.csv"), DataFrame)

# Adjust CUDA settings to make sure no scalar calculations occur
CUDA.allowscalar(false)

# Set up symbolic variables for the problem
Symbolics.@variables a0, a1, a2, wval, Tval

# Represent parts of the polynomial expression using all_evaluators from SCMC
f1_lo, f1_hi, f1_cv, f1_cc, order1 = all_evaluators(a0 + a1*wval + a2*wval^2)
f2_lo, f2_hi, f2_cv, f2_cc, order2 = all_evaluators(a0 + (1/Tval)*a1)
f3_lo, f3_hi, f3_cv, f3_cc, order3 = all_evaluators(exp(a0))
Symbolics.@variables calc, data

# Use only the convex_evaluator function to get the convex relaxation for the SSE
SSE_cv, _ = convex_evaluator(((calc - data)/data)^2)

# Set up the GPU-compatible L2norm calculation
L2norm(p...) = L2norm(pressure_data, p...)
function L2norm(pressure_data::DataFrame, p...)
    pset = pressure_data.pressure
    wset = pressure_data.w
    Tset = pressure_data.T
    
    L = length(p[1])
    SSEcv = CUDA.zeros(Float64, L)

    temp1_cc = CUDA.zeros(Float64, L)
    temp1_cv = CUDA.zeros(Float64, L)
    temp1_hi = CUDA.zeros(Float64, L)
    temp1_lo = CUDA.zeros(Float64, L)
    
    temp2_cc = CUDA.zeros(Float64, L)
    temp2_cv = CUDA.zeros(Float64, L)
    temp2_hi = CUDA.zeros(Float64, L)
    temp2_lo = CUDA.zeros(Float64, L)
    
    temp3_cc = CUDA.zeros(Float64, L)
    temp3_cv = CUDA.zeros(Float64, L)
    temp3_hi = CUDA.zeros(Float64, L)
    temp3_lo = CUDA.zeros(Float64, L)

    for i = 1:length(pset)
        temp1_cc .= f1_cc.(p[1:12]..., wset[i], wset[i], wset[i], wset[i])
        temp1_cv .= f1_cv.(p[1:12]..., wset[i], wset[i], wset[i], wset[i])
        temp1_hi .= f1_hi.(p[1:12]..., wset[i], wset[i], wset[i], wset[i])
        temp1_lo .= f1_lo.(p[1:12]..., wset[i], wset[i], wset[i], wset[i])

        temp2_cc .= f1_cc.(p[13:24]..., wset[i], wset[i], wset[i], wset[i])
        temp2_cv .= f1_cv.(p[13:24]..., wset[i], wset[i], wset[i], wset[i])
        temp2_hi .= f1_hi.(p[13:24]..., wset[i], wset[i], wset[i], wset[i])
        temp2_lo .= f1_lo.(p[13:24]..., wset[i], wset[i], wset[i], wset[i])

        temp3_cc .= f2_cc.(Tset[i], Tset[i], Tset[i], Tset[i],
                          temp1_cc, temp1_cv, temp1_hi, temp1_lo, 
                          temp2_cc, temp2_cv, temp2_hi, temp2_lo)
        temp3_cv .= f2_cv.(Tset[i], Tset[i], Tset[i], Tset[i],
                          temp1_cc, temp1_cv, temp1_hi, temp1_lo, 
                          temp2_cc, temp2_cv, temp2_hi, temp2_lo)
        temp3_hi .= f2_hi.(Tset[i], Tset[i], Tset[i], Tset[i],
                          temp1_cc, temp1_cv, temp1_hi, temp1_lo, 
                          temp2_cc, temp2_cv, temp2_hi, temp2_lo)
        temp3_lo .= f2_lo.(Tset[i], Tset[i], Tset[i], Tset[i],
                          temp1_cc, temp1_cv, temp1_hi, temp1_lo, 
                          temp2_cc, temp2_cv, temp2_hi, temp2_lo)

        temp1_cc .= f3_cc.(temp3_cc, temp3_cv, temp3_hi, temp3_lo)
        temp1_cv .= f3_cv.(temp3_cc, temp3_cv, temp3_hi, temp3_lo)
        temp1_hi .= f3_hi.(temp3_cc, temp3_cv, temp3_hi, temp3_lo)
        temp1_lo .= f3_lo.(temp3_cc, temp3_cv, temp3_hi, temp3_lo)

        SSEcv .+= SSE_cv.(temp1_cc, temp1_cv, temp1_hi, temp1_lo, pset[i], pset[i], pset[i], pset[i])
    end
    return SSEcv
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
factory = () -> EAGO.Optimizer(SubSolvers(; t = ExtendGPU(L2norm, 6)))
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
factory = () -> EAGO.Optimizer(SubSolvers(; t = ExtendGPU(L2norm, 6, alpha=0.00002)))
opt = optimizer_with_attributes(factory, "enable_optimize_hook" => true,
                                         "branch_variable" => Bool[true for i in 1:6],
                                         "force_global_solve" => true,
                                         "node_limit" => Int(3e8),
                                         "time_limit" => 7200.0,
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

