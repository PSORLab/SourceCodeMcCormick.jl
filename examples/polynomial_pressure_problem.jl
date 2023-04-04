
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
function L2norm(p...)
    pset = [12.38, 10.10, 6.31, 28.33, 24.17, 18.00, 12.38, 60.13, 48.49, 36.46, 26.54, 20.32, 13.61, 
            118.57, 92.90, 70.48, 53.49, 40.23, 27.84, 215.95, 179.25, 126.59, 100.39, 78.16, 52.82, 
            33.67, 377.13, 292.92, 220.38, 175.32, 140.91, 93.25, 59.32, 38.36, 596.57, 496.87, 361.58,
            288.22, 229.18, 159.01, 101.20, 63.43, 920.02, 789.60, 564.39, 451.19, 357.82, 249.55, 162.51, 99.24]
    wset = [0.4992, 0.5991, 0.6997, 0.4994, 0.6000, 0.7000, 0.7493, 0.4997, 0.6006, 0.7003, 0.7495, 
            0.8000, 0.8503, 0.5004, 0.6015, 0.7008, 0.7500, 0.8004, 0.8506, 0.5014, 0.6033, 0.7017,
            0.7508, 0.8011, 0.8511, 0.8998, 0.5030, 0.6054, 0.7031, 0.7520, 0.8022, 0.8519, 0.9003, 
            0.9501, 0.5050, 0.6092, 0.7051, 0.7537, 0.8036, 0.8531, 0.9011, 0.9506, 0.5079, 0.6145,
            0.7078, 0.7560, 0.8056, 0.8547, 0.9022, 0.9512]
    Tset = [333.15, 333.15, 333.15, 353.15, 353.15, 353.15, 353.15, 373.15, 373.15, 373.15, 373.15, 
            373.15, 373.15, 393.15, 393.15, 393.15, 393.15, 393.15, 393.15, 413.15, 413.15, 413.15, 
            413.15, 413.15, 413.15, 413.15, 433.15, 433.15, 433.15, 433.15, 433.15, 433.15, 433.15, 
            433.15, 453.15, 453.15, 453.15, 453.15, 453.15, 453.15, 453.15, 453.15, 473.15, 473.15, 
            473.15, 473.15, 473.15, 473.15, 473.15, 473.15]
    
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
function L2norm_cpu(p1::T, p2::T, p3::T, p4::T, p5::T, p6::T) where {T<:Real}
    SSE = zero(T)
    pset = [12.38, 10.10, 6.31, 28.33, 24.17, 18.00, 12.38, 60.13, 48.49, 36.46, 26.54, 20.32, 13.61, 
            118.57, 92.90, 70.48, 53.49, 40.23, 27.84, 215.95, 179.25, 126.59, 100.39, 78.16, 52.82, 
            33.67, 377.13, 292.92, 220.38, 175.32, 140.91, 93.25, 59.32, 38.36, 596.57, 496.87, 361.58,
            288.22, 229.18, 159.01, 101.20, 63.43, 920.02, 789.60, 564.39, 451.19, 357.82, 249.55, 162.51, 99.24]
    wset = [0.4992, 0.5991, 0.6997, 0.4994, 0.6000, 0.7000, 0.7493, 0.4997, 0.6006, 0.7003, 0.7495, 
            0.8000, 0.8503, 0.5004, 0.6015, 0.7008, 0.7500, 0.8004, 0.8506, 0.5014, 0.6033, 0.7017,
            0.7508, 0.8011, 0.8511, 0.8998, 0.5030, 0.6054, 0.7031, 0.7520, 0.8022, 0.8519, 0.9003, 
            0.9501, 0.5050, 0.6092, 0.7051, 0.7537, 0.8036, 0.8531, 0.9011, 0.9506, 0.5079, 0.6145,
            0.7078, 0.7560, 0.8056, 0.8547, 0.9022, 0.9512]
    Tset = [333.15, 333.15, 333.15, 353.15, 353.15, 353.15, 353.15, 373.15, 373.15, 373.15, 373.15, 
            373.15, 373.15, 393.15, 393.15, 393.15, 393.15, 393.15, 393.15, 413.15, 413.15, 413.15, 
            413.15, 413.15, 413.15, 413.15, 433.15, 433.15, 433.15, 433.15, 433.15, 433.15, 433.15, 
            433.15, 453.15, 453.15, 453.15, 453.15, 453.15, 453.15, 453.15, 453.15, 473.15, 473.15, 
            473.15, 473.15, 473.15, 473.15, 473.15, 473.15]

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

