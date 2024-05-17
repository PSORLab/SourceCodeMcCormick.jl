
#############################################################################
# This code uses the GPU to speed up the solve time for the kinetic
# optimization example described in Mitsos2009, Stuber2014, Wilhelm2019, and 
# Wilhelm2020. Specifically, this file is solving the explicit Euler
# formulation of the problem.
#############################################################################

# Import the necessary packages
using EAGO, JuMP, CSV, DataFrames, Symbolics, SourceCodeMcCormick, CUDA

# Import the ParBB algorithm
include(joinpath(@__DIR__, "ParBB", "extension.jl"))
include(joinpath(@__DIR__, "ParBB", "subroutines.jl"))

# Import the kinetic intensity data
data = CSV.read(joinpath(@__DIR__, "kinetic_intensity_data.csv"), DataFrame)
bounds = CSV.read(joinpath(@__DIR__, "implicit_variable_bounds.csv"), DataFrame)

# Define the constant terms in the expressions
T = 273.0
K2 = 46.0*exp(6500.0/T - 18.0)
K3 = 2.0*K2
k1 = 53.0
k1s = k1*10^(-6)
k5 = 0.0012
cO2 = 0.002
h = 0.01

# Using Symbolics.jl, declare variables that go into our expressions
Symbolics.@variables p[1:3], xA, xB, xD, xY, xZ

# Using SourceCodeMcCormick.jl's "all_evaluators" function, generate
# sets of evaluation functions. The inputs to "all_evaluators" are the
# equations that define the dynamics of the explicit Euler formulation.
# The "eval" objects are functions that return evaluated bounds and
# relaxations of the provided function, with the required inputs listed
# in "orderA", "orderB", etc. 
term1 = k1*xY*xZ-cO2*(p[1]+p[2])*xA
term2 = p[1]*xD/K2+p[2]*xB/K3-k5*xA^2
xA_lo_eval, xA_hi_eval, xA_cv_eval, xA_cc_eval, orderA = all_evaluators(xA + h*(term1 + term2))
xB_lo_eval, xB_hi_eval, xB_cv_eval, xB_cc_eval, orderB = all_evaluators(xB + h*(p[2]*cO2*xA-(p[2]/K3+p[3])*xB))
xD_lo_eval, xD_hi_eval, xD_cv_eval, xD_cc_eval, orderD = all_evaluators(xD + h*(p[1]*cO2*xA-p[1]*xD/K2))
xY_lo_eval, xY_hi_eval, xY_cv_eval, xY_cc_eval, orderY = all_evaluators(xY + h*(-k1s*xY*xZ))
xZ_lo_eval, xZ_hi_eval, xZ_cv_eval, xZ_cc_eval, orderZ = all_evaluators(xZ + h*(-k1*xY*xZ))

# For the objective function we are calculating the SSE. At each time point
# we will calculate only the convex relaxation, and this will be summed over
# all time points (sum of convex relaxations of operands is the convex relaxation
# of the sum). We only need the convex evaluation, so we use "convex_evaluator"
# instead of "all_evaluators"
I(x,y,z) = x + (2/21)*y + (2/21)*z
Symbolics.@variables data_point # Data is represented symbolically since this
                                # is part of what's being squared in the SSE calculation
SSE_convex, SSE_order= convex_evaluator((I(xA, xB, xD) - data_point)^2)
SSE_lo, SSE_hi, SSE_cv, SSE_cc, SSE_order_all = all_evaluators((I(xA, xB, xD) - data_point)^2)


explicit_euler_gpu64(p_1_cc, p_1_cv, p_1_hi, p_1_lo, 
                   p_2_cc, p_2_cv, p_2_hi, p_2_lo, 
                   p_3_cc, p_3_cv, p_3_hi, p_3_lo) = 
    explicit_euler_gpu64(p_1_cc, p_1_cv, p_1_hi, p_1_lo, 
                       p_2_cc, p_2_cv, p_2_hi, p_2_lo, 
                       p_3_cc, p_3_cv, p_3_hi, p_3_lo, data)      

# And now the actual function
function explicit_euler_gpu64(p_1_cc::T, p_1_cv::T, p_1_hi::T, p_1_lo::T, 
                            p_2_cc::T, p_2_cv::T, p_2_hi::T, p_2_lo::T, 
                            p_3_cc::T, p_3_cv::T, p_3_hi::T, p_3_lo::T, data) where T<:CuArray
    
    L = length(p_1_lo)

    # We've taken in all the p information. Before the loop, initialize
    # all the x variables, assuming we're dealing with CuArrays
    xA_lo = CUDA.zeros(Float64, L)
    xA_hi = CUDA.zeros(Float64, L) # Starting bounds: [0.0, 140.0]
    xA_cv = CUDA.zeros(Float64, L)
    xA_cc = CUDA.zeros(Float64, L)
    xB_lo = CUDA.zeros(Float64, L)
    xB_hi = CUDA.zeros(Float64, L) # Starting bounds: [0.0, 140.0]
    xB_cv = CUDA.zeros(Float64, L)
    xB_cc = CUDA.zeros(Float64, L)
    xD_lo = CUDA.zeros(Float64, L) # Starting bounds: [0.0, 140.0]
    xD_hi = CUDA.zeros(Float64, L)
    xD_cv = CUDA.zeros(Float64, L)
    xD_cc = CUDA.zeros(Float64, L)
    xY = CUDA.fill(0.4, L) #Starting bounds: [0.0, 0.4]
    xZ = CUDA.fill(140.0, L) #Starting bounds: [0.0, 140.0]

    # Set up the "new" ones
    xA_lo_new = CuArray{Float64}(undef, L)
    xA_hi_new = CuArray{Float64}(undef, L)
    xA_cv_new = CuArray{Float64}(undef, L)
    xA_cc_new = CuArray{Float64}(undef, L)
    xB_lo_new = CuArray{Float64}(undef, L)
    xB_hi_new = CuArray{Float64}(undef, L)
    xB_cv_new = CuArray{Float64}(undef, L)
    xB_cc_new = CuArray{Float64}(undef, L)
    xD_lo_new = CuArray{Float64}(undef, L)
    xD_hi_new = CuArray{Float64}(undef, L)
    xD_cv_new = CuArray{Float64}(undef, L)
    xD_cc_new = CuArray{Float64}(undef, L)
    xY_new = CuArray{Float64}(undef, L)
    xZ_new = CuArray{Float64}(undef, L)

    # Set up SSE and data point placeholder to be 0
    SSE = CUDA.zeros(Float64, L)
    point = CUDA.zeros(Float64, L)

    # Now that those are initialized, we can begin our loop.
    # Calculate all the terms for x's
    for i = 1:200
        xA_lo_new .= xA_lo_eval.(p_1_cc, p_1_cv, p_1_hi, p_1_lo, 
                                 p_2_cc, p_2_cv, p_2_hi, p_2_lo,
                                 xA_cc, xA_cv, xA_hi, xA_lo,
                                 xB_cc, xB_cv, xB_hi, xB_lo,
                                 xD_cc, xD_cv, xD_hi, xD_lo,
                                 xY, xY, xY, xY,
                                 xZ, xZ, xZ, xZ)
        xA_hi_new .= xA_hi_eval.(p_1_cc, p_1_cv, p_1_hi, p_1_lo, 
                                 p_2_cc, p_2_cv, p_2_hi, p_2_lo,
                                 xA_cc, xA_cv, xA_hi, xA_lo,
                                 xB_cc, xB_cv, xB_hi, xB_lo,
                                 xD_cc, xD_cv, xD_hi, xD_lo,
                                 xY, xY, xY, xY,
                                 xZ, xZ, xZ, xZ)
        xA_cv_new .= xA_cv_eval.(p_1_cc, p_1_cv, p_1_hi, p_1_lo, 
                                 p_2_cc, p_2_cv, p_2_hi, p_2_lo,
                                 xA_cc, xA_cv, xA_hi, xA_lo,
                                 xB_cc, xB_cv, xB_hi, xB_lo,
                                 xD_cc, xD_cv, xD_hi, xD_lo,
                                 xY, xY, xY, xY,
                                 xZ, xZ, xZ, xZ)
        xA_cc_new .= xA_cc_eval.(p_1_cc, p_1_cv, p_1_hi, p_1_lo, 
                                 p_2_cc, p_2_cv, p_2_hi, p_2_lo,
                                 xA_cc, xA_cv, xA_hi, xA_lo,
                                 xB_cc, xB_cv, xB_hi, xB_lo,
                                 xD_cc, xD_cv, xD_hi, xD_lo,
                                 xY, xY, xY, xY,
                                 xZ, xZ, xZ, xZ)
        xB_lo_new .= xB_lo_eval.(p_2_cc, p_2_cv, p_2_hi, p_2_lo, 
                                 p_3_cc, p_3_cv, p_3_hi, p_3_lo,
                                 xA_cc, xA_cv, xA_hi, xA_lo,
                                 xB_cc, xB_cv, xB_hi, xB_lo)
        xB_hi_new .= xB_hi_eval.(p_2_cc, p_2_cv, p_2_hi, p_2_lo, 
                                 p_3_cc, p_3_cv, p_3_hi, p_3_lo,
                                 xA_cc, xA_cv, xA_hi, xA_lo,
                                 xB_cc, xB_cv, xB_hi, xB_lo)
        xB_cv_new .= xB_cv_eval.(p_2_cc, p_2_cv, p_2_hi, p_2_lo, 
                                 p_3_cc, p_3_cv, p_3_hi, p_3_lo,
                                 xA_cc, xA_cv, xA_hi, xA_lo,
                                 xB_cc, xB_cv, xB_hi, xB_lo)
        xB_cc_new .= xB_cc_eval.(p_2_cc, p_2_cv, p_2_hi, p_2_lo, 
                                 p_3_cc, p_3_cv, p_3_hi, p_3_lo,
                                 xA_cc, xA_cv, xA_hi, xA_lo,
                                 xB_cc, xB_cv, xB_hi, xB_lo)
        xD_lo_new .= xD_lo_eval.(p_1_cc, p_1_cv, p_1_hi, p_1_lo,
                                 xA_cc, xA_cv, xA_hi, xA_lo,
                                 xD_cc, xD_cv, xD_hi, xD_lo)
        xD_hi_new .= xD_hi_eval.(p_1_cc, p_1_cv, p_1_hi, p_1_lo,
                                 xA_cc, xA_cv, xA_hi, xA_lo,
                                 xD_cc, xD_cv, xD_hi, xD_lo)
        xD_cv_new .= xD_cv_eval.(p_1_cc, p_1_cv, p_1_hi, p_1_lo,
                                 xA_cc, xA_cv, xA_hi, xA_lo,
                                 xD_cc, xD_cv, xD_hi, xD_lo)
        xD_cc_new .= xD_cc_eval.(p_1_cc, p_1_cv, p_1_hi, p_1_lo,
                                 xA_cc, xA_cv, xA_hi, xA_lo,
                                 xD_cc, xD_cv, xD_hi, xD_lo)
        xY_new .= xY .+ h.*(-k1s.*xY.*xZ)
        xZ_new .= xZ .+ h.*(-k1.*xY.*xZ)

        # Plug in these values to the SSE equation
        point = CUDA.fill(data[!, :intensity][i], L)
        SSE .+= SSE_convex.(point, point, point, point,
                           xA_cc_new, xA_cv_new, xA_hi_new, xA_lo_new,
                           xB_cc_new, xB_cv_new, xB_hi_new, xB_lo_new,
                           xD_cc_new, xD_cv_new, xD_hi_new, xD_lo_new)

        # Convert all the "olds" to their "new" values
        xA_lo .= xA_lo_new; xA_hi .= xA_hi_new; xA_cv .= xA_cv_new; xA_cc .= xA_cc_new
        xB_lo .= xB_lo_new; xB_hi .= xB_hi_new; xB_cv .= xB_cv_new; xB_cc .= xB_cc_new
        xD_lo .= xD_lo_new; xD_hi .= xD_hi_new; xD_cv .= xD_cv_new; xD_cc .= xD_cc_new
        xY .= xY_new
        xZ .= xZ_new
    end

    # SSE is now the convex relaxation of the SSE
    return SSE
end

function cpu_euler(p)
    x = zeros(typeof(p[1]), 1005);
    x[4] = 0.4; x[5] = 140
    T = 273; delT = 0.01; cO2 = 2e-3; k1 = 53; k1s = k1*1E-6;
    K2 = 46*exp(6500/T-18); K3 = 2*K2; h = delT; k5 = 1.2E-3

    for i = 1:200
        term1 = k1*x[5i-1]*x[5i]-cO2*(p[1]+p[2])*x[5i-4]
        term2 = p[1]*x[5i-2]/K2+p[2]*x[5i-3]/K3-k5*x[5i-4]^2
        x[5i+1] = x[5i-4] + h*(term1 + term2)
        x[5i+2] = x[5i-3] + h*(p[2]*cO2*x[5i-4]-(p[2]/K3+p[3])*x[5i-3])
        x[5i+3] = x[5i-2] + h*(p[1]*cO2*x[5i-4]-p[1]*x[5i-2]/K2)
        x[5i+4] = x[5i-1] + h*(-k1s*x[5i-1]*x[5i])
        x[5i+5] = x[5i] + h*(-k1*x[5i-1]*x[5i])
    end
    return x
end

function objective(p...)
    x = cpu_euler(p)
    SSE = zero(typeof(p[1]))
    for i = 1:200
        SSE += (I(x[5i+1],x[5i+2],x[5i+3]) - data[!, :intensity][i])^2
    end
    return SSE
end


# Compilation run
factory = () -> EAGO.Optimizer(SubSolvers(; t = ExtendGPU(explicit_euler_gpu64, 3)))
opt = optimizer_with_attributes(factory, "enable_optimize_hook" => true,
                                         "branch_variable" => Bool[true for i in 1:3],
                                         "force_global_solve" => true,
                                         "node_limit" => Int(3e8),
                                         "time_limit" => 10.0,
                                         "output_iterations" => 10000)
m = Model(opt)
pL = [10.0, 10.0, 0.001]
pU = [1200.0, 1200.0, 40.0]
@variable(m, pL[i] <= p[i=1:3] <= pU[i])

fobj(p...) = objective(p...)
JuMP.register(m, :fobj, 3, fobj, autodiff=true)
@NLobjective(m, Min, fobj(p...))

optimize!(m)


# Run for non-compilation timing
factory = () -> EAGO.Optimizer(SubSolvers(; t = ExtendGPU(explicit_euler_gpu64, 3, alpha=0.00002)))
opt = optimizer_with_attributes(factory, "enable_optimize_hook" => true,
                                         "branch_variable" => Bool[true for i in 1:3],
                                         "force_global_solve" => true,
                                         "node_limit" => Int(3e8),
                                         "time_limit" => 7200.0,
                                         "output_iterations" => 10,
                                         "log_on" => true)
m = Model(opt)
pL = [10.0, 10.0, 0.001]
pU = [1200.0, 1200.0, 40.0]
@variable(m, pL[i] <= p[i=1:3] <= pU[i])

fobj(p...) = objective(p...)
JuMP.register(m, :fobj, 3, fobj, autodiff=true)
@NLobjective(m, Min, fobj(p...))

optimize!(m)
