
# This file contains testable versions of kernels that accept matrices for each
# variable, with elements in the order: [cv, cc, lo, hi, cvgrad, ccgrad].
# These kernels do not get used within `kgen`, but should be essentially the same
# as what gets written to the generated kernels. "string_math_kernels.jl" contains
# these same functions, but in buffer/string form for the purposes of writing
# new kernels.

#=
Unitary Rules
=#
# Addition with a constant
# max threads: 1024
function SCMC_cadd_kernel(OUT::CuDeviceMatrix, CONST::Real, x::CuDeviceMatrix)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    colmax = Int32((size(OUT,2)-4)/2)

    while idx <= Int32(size(OUT,1))
        # Reset the column counter
        col = Int32(1)

        # Addition
        OUT[idx,3] = x[idx,3] + CONST
        OUT[idx,4] = x[idx,4] + CONST
        OUT[idx,1] = x[idx,1] + CONST
        OUT[idx,2] = x[idx,2] + CONST

        # Perform the cut operation
        if OUT[idx,1] < OUT[idx,3]
            OUT[idx,1] = OUT[idx,3]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-2*colmax+col] = 0.0
                col += Int32(1)
            end
        end
        if OUT[idx,2] > OUT[idx,4]
            OUT[idx,2] = OUT[idx,4]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-1*colmax+col] = 0.0
                col += Int32(1)
            end
        end

        idx += stride
    end
    return nothing
end

# Addition to an existing variable
# max threads: 1024
function SCMC_add_to_kernel(OUT::CuDeviceMatrix, x::CuDeviceMatrix)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    colmax = Int32((size(OUT,2)-4)/2)

    while idx <= Int32(size(OUT,1))
        # Reset the column counter
        col = Int32(1)

        # Negation
        OUT[idx,3] += x[idx,3]
        OUT[idx,4] += x[idx,4]
        OUT[idx,1] += x[idx,1]
        OUT[idx,2] += x[idx,2]
        while col <= colmax
            OUT[idx,end-2*colmax+col] += x[idx,end-2*colmax+col]
            OUT[idx,end-1*colmax+col] += x[idx,end-1*colmax+col]
            col += Int32(1)
        end

        # Perform the cut operation
        if OUT[idx,1] < OUT[idx,3]
            OUT[idx,1] = OUT[idx,3]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-2*colmax+col] = 0.0
                col += Int32(1)
            end
        end
        if OUT[idx,2] > OUT[idx,4]
            OUT[idx,2] = OUT[idx,4]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-1*colmax+col] = 0.0
                col += Int32(1)
            end
        end

        idx += stride
    end
    return nothing
end

# Negating (equivalent to multiplication by a constant if constant==-1)
# max threads: 1024
function SCMC_negate_kernel(OUT::CuDeviceMatrix, x::CuDeviceMatrix)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    colmax = Int32((size(OUT,2)-4)/2)

    while idx <= Int32(size(OUT,1))
        # Reset the column counter
        col = Int32(1)

        # Negation
        OUT[idx,3] = -x[idx,4]
        OUT[idx,4] = -x[idx,3]
        OUT[idx,1] = -x[idx,2]
        OUT[idx,2] = -x[idx,1]
        while col <= colmax
            OUT[idx,end-2*colmax+col] = -x[idx,end-1*colmax+col]
            OUT[idx,end-1*colmax+col] = -x[idx,end-2*colmax+col]
            col += Int32(1)
        end

        # Perform the cut operation
        if OUT[idx,1] < OUT[idx,3]
            OUT[idx,1] = OUT[idx,3]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-2*colmax+col] = 0.0
                col += Int32(1)
            end
        end
        if OUT[idx,2] > OUT[idx,4]
            OUT[idx,2] = OUT[idx,4]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-1*colmax+col] = 0.0
                col += Int32(1)
            end
        end

        idx += stride
    end
    return nothing
end

# Exponential
# max threads: 896
function SCMC_exp_kernel(OUT::CuDeviceMatrix, x::CuDeviceMatrix)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    colmax = Int32((size(OUT,2)-4)/2)

    while idx <= Int32(size(OUT,1))
        # Reset the column counter
        col = Int32(1)

        # Begin exponential
        OUT[idx,3] = exp(x[idx,3])
        OUT[idx,4] = exp(x[idx,4])
        if x[idx,2] >= x[idx,1]
            if x[idx,1] >= x[idx,4]
                if x[idx,1] >= x[idx,3]
                    OUT[idx,1] = exp(x[idx,1])
                    OUT[idx,2] = (exp(x[idx,3])*(x[idx,4] - x[idx,1]) + exp(x[idx,4])*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = exp(x[idx,1])*x[idx,end-2*colmax+col]
                        OUT[idx,end-1*colmax+col] = (exp(x[idx,4]) - exp(x[idx,3]))/(x[idx,4] - x[idx,3])*x[idx,end-2*colmax+col]
                        col += Int32(1)
                    end
                elseif x[idx,2] == x[idx,1]
                    OUT[idx,1] = exp(x[idx,1])
                    OUT[idx,2] = (exp(x[idx,3])*(x[idx,4] - x[idx,1]) + exp(x[idx,4])*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = exp(x[idx,1])*x[idx,end-2*colmax+col]
                        OUT[idx,end-1*colmax+col] = (exp(x[idx,4]) - exp(x[idx,3]))/(x[idx,4] - x[idx,3])*x[idx,end-2*colmax+col]
                        col += Int32(1)
                    end
                elseif x[idx,3] >= x[idx,2]
                    OUT[idx,1] = exp(x[idx,2])
                    OUT[idx,2] = (exp(x[idx,3])*(x[idx,4] - x[idx,1]) + exp(x[idx,4])*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = exp(x[idx,2])*x[idx,end-1*colmax+col]
                        OUT[idx,end-1*colmax+col] = (exp(x[idx,4]) - exp(x[idx,3]))/(x[idx,4] - x[idx,3])*x[idx,end-2*colmax+col]
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = exp(x[idx,3])
                    OUT[idx,2] = (exp(x[idx,3])*(x[idx,4] - x[idx,1]) + exp(x[idx,4])*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = 0.0
                        OUT[idx,end-1*colmax+col] = (exp(x[idx,4]) - exp(x[idx,3]))/(x[idx,4] - x[idx,3])*x[idx,end-2*colmax+col]
                        col += Int32(1)
                    end
                end
            elseif x[idx,2] == x[idx,1]
                OUT[idx,1] = exp(x[idx,1])
                OUT[idx,2] = (exp(x[idx,3])*(x[idx,4] - x[idx,1]) + exp(x[idx,4])*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                while col <= colmax
                    OUT[idx,end-2*colmax+col] = exp(x[idx,1])*x[idx,end-2*colmax+col]
                    OUT[idx,end-1*colmax+col] = (exp(x[idx,4]) - exp(x[idx,3]))/(x[idx,4] - x[idx,3])*x[idx,end-2*colmax+col]
                    col += Int32(1)
                end
            elseif x[idx,4] >= x[idx,2]
                if x[idx,1] >= x[idx,3]
                    OUT[idx,1] = exp(x[idx,1])
                    OUT[idx,2] = (exp(x[idx,3])*(x[idx,4] - x[idx,2]) + exp(x[idx,4])*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = exp(x[idx,1])*x[idx,end-2*colmax+col]
                        OUT[idx,end-1*colmax+col] = (exp(x[idx,4]) - exp(x[idx,3]))/(x[idx,4] - x[idx,3])*x[idx,end-1*colmax+col]
                        col += Int32(1)
                    end
                elseif x[idx,3] >= x[idx,2]
                    OUT[idx,1] = exp(x[idx,2])
                    OUT[idx,2] = (exp(x[idx,3])*(x[idx,4] - x[idx,2]) + exp(x[idx,4])*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = exp(x[idx,2])*x[idx,end-1*colmax+col]
                        OUT[idx,end-1*colmax+col] = (exp(x[idx,4]) - exp(x[idx,3]))/(x[idx,4] - x[idx,3])*x[idx,end-1*colmax+col]
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = exp(x[idx,3])
                    OUT[idx,2] = (exp(x[idx,3])*(x[idx,4] - x[idx,2]) + exp(x[idx,4])*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = 0.0
                        OUT[idx,end-1*colmax+col] = (exp(x[idx,4]) - exp(x[idx,3]))/(x[idx,4] - x[idx,3])*x[idx,end-1*colmax+col]
                        col += Int32(1)
                    end
                end
            else
                if x[idx,1] >= x[idx,3]
                    OUT[idx,1] = exp(x[idx,1])
                    OUT[idx,2] = exp(x[idx,4])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = exp(x[idx,1])*x[idx,end-2*colmax+col]
                        OUT[idx,end-1*colmax+col] = 0.0
                        col += Int32(1)
                    end
                elseif x[idx,3] >= x[idx,2]
                    OUT[idx,1] = exp(x[idx,2])
                    OUT[idx,2] = exp(x[idx,4])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = exp(x[idx,2])*x[idx,end-1*colmax+col]
                        OUT[idx,end-1*colmax+col] = 0.0
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = exp(x[idx,3])
                    OUT[idx,2] = exp(x[idx,4])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = 0.0
                        OUT[idx,end-1*colmax+col] = 0.0
                        col += Int32(1)
                    end
                end
            end
        elseif x[idx,4] >= x[idx,1]
            if x[idx,3] >= x[idx,1]
                OUT[idx,1] = exp(x[idx,1])
                OUT[idx,2] = (exp(x[idx,3])*(x[idx,4] - x[idx,1]) + exp(x[idx,4])*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                while col <= colmax
                    OUT[idx,end-2*colmax+col] = exp(x[idx,1])*x[idx,end-2*colmax+col]
                    OUT[idx,end-1*colmax+col] = (exp(x[idx,4]) - exp(x[idx,3]))/(x[idx,4] - x[idx,3])*x[idx,end-2*colmax+col]
                    col += Int32(1)
                end
            elseif x[idx,2] >= x[idx,3]
                OUT[idx,1] = exp(x[idx,2])
                OUT[idx,2] = (exp(x[idx,3])*(x[idx,4] - x[idx,1]) + exp(x[idx,4])*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                while col <= colmax
                    OUT[idx,end-2*colmax+col] = exp(x[idx,2])*x[idx,end-1*colmax+col]
                    OUT[idx,end-1*colmax+col] = (exp(x[idx,4]) - exp(x[idx,3]))/(x[idx,4] - x[idx,3])*x[idx,end-2*colmax+col]
                    col += Int32(1)
                end
            else
                OUT[idx,1] = exp(x[idx,3])
                OUT[idx,2] = (exp(x[idx,3])*(x[idx,4] - x[idx,1]) + exp(x[idx,4])*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                while col <= colmax
                    OUT[idx,end-2*colmax+col] = 0.0
                    OUT[idx,end-1*colmax+col] = (exp(x[idx,4]) - exp(x[idx,3]))/(x[idx,4] - x[idx,3])*x[idx,end-2*colmax+col]
                    col += Int32(1)
                end
            end
        elseif x[idx,2] >= x[idx,4]
            if x[idx,3] >= x[idx,1]
                OUT[idx,1] = exp(x[idx,1])
                OUT[idx,2] = (exp(x[idx,3])*(x[idx,4] - x[idx,2]) + exp(x[idx,4])*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                while col <= colmax
                    OUT[idx,end-2*colmax+col] = exp(x[idx,1])*x[idx,end-2*colmax+col]
                    OUT[idx,end-1*colmax+col] = (exp(x[idx,4]) - exp(x[idx,3]))/(x[idx,4] - x[idx,3])*x[idx,end-1*colmax+col]
                    col += Int32(1)
                end
            elseif x[idx,2] >= x[idx,3]
                OUT[idx,1] = exp(x[idx,2])
                OUT[idx,2] = (exp(x[idx,3])*(x[idx,4] - x[idx,2]) + exp(x[idx,4])*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                while col <= colmax
                    OUT[idx,end-2*colmax+col] = exp(x[idx,2])*x[idx,end-1*colmax+col]
                    OUT[idx,end-1*colmax+col] = (exp(x[idx,4]) - exp(x[idx,3]))/(x[idx,4] - x[idx,3])*x[idx,end-1*colmax+col]
                    col += Int32(1)
                end
            else
                OUT[idx,1] = exp(x[idx,3])
                OUT[idx,2] = (exp(x[idx,3])*(x[idx,4] - x[idx,2]) + exp(x[idx,4])*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                while col <= colmax
                    OUT[idx,end-2*colmax+col] = 0.0
                    OUT[idx,end-1*colmax+col] = (exp(x[idx,4]) - exp(x[idx,3]))/(x[idx,4] - x[idx,3])*x[idx,end-1*colmax+col]
                    col += Int32(1)
                end
            end
        else
            if x[idx,3] >= x[idx,1]
                OUT[idx,1] = exp(x[idx,1])
                OUT[idx,2] = exp(x[idx,4])
                while col <= colmax
                    OUT[idx,end-2*colmax+col] = exp(x[idx,1])*x[idx,end-2*colmax+col]
                    OUT[idx,end-1*colmax+col] = 0.0
                    col += Int32(1)
                end
            elseif x[idx,2] >= x[idx,3]
                OUT[idx,1] = exp(x[idx,2])
                OUT[idx,2] = exp(x[idx,4])
                while col <= colmax
                    OUT[idx,end-2*colmax+col] = exp(x[idx,2])*x[idx,end-1*colmax+col]
                    OUT[idx,end-1*colmax+col] = 0.0
                    col += Int32(1)
                end
            else
                OUT[idx,1] = exp(x[idx,3])
                OUT[idx,2] = exp(x[idx,4])
                while col <= colmax
                    OUT[idx,end-2*colmax+col] = 0.0
                    OUT[idx,end-1*colmax+col] = 0.0
                    col += Int32(1)
                end
            end
        end
        
        # Check if x is degenerate, and if so, set the concave relaxation subgradient to 0.0
        if x[idx,4] == x[idx,3]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-1*colmax+col] = 0.0
                col += Int32(1)
            end
        end

        # Perform the cut operation
        if OUT[idx,1] < OUT[idx,3]
            OUT[idx,1] = OUT[idx,3]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-2*colmax+col] = 0.0
                col += Int32(1)
            end
        end
        if OUT[idx,2] > OUT[idx,4]
            OUT[idx,2] = OUT[idx,4]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-1*colmax+col] = 0.0
                col += Int32(1)
            end
        end

        idx += stride
    end
    return nothing
end

# Logarithm
# max threads: 896
function SCMC_log_kernel(OUT::CuDeviceMatrix, x::CuDeviceMatrix)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    colmax = Int32((size(OUT,2)-4)/2)

    while idx <= Int32(size(OUT,1))
        # Reset the column counter
        col = Int32(1)

        # Logarithm
        OUT[idx,3] = log(max(x[idx,3], 0.0))
        OUT[idx,4] = log(max(x[idx,4], 0.0))
        if x[idx,2] >= x[idx,1]
            if x[idx,1] >= x[idx,4]
                if x[idx,1] >= x[idx,3]
                    if x[idx,4] > x[idx,3]
                        OUT[idx,1] = (log(max(x[idx,4], 0.0)) - log(max(x[idx,3], 0.0)))/(x[idx,4] - x[idx,3]) * (x[idx,1] - x[idx,3]) + log(max(x[idx,3], 0.0))
                        OUT[idx,2] = log(max(x[idx,1], 0.0))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * (log(max(x[idx,4], 0.0)) - log(max(x[idx,3], 0.0)))/(x[idx,4] - x[idx,3])
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (1/x[idx,1])
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = log(max(x[idx,3], 0.0))
                        OUT[idx,2] = log(max(x[idx,1], 0.0))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (1/x[idx,1])
                            col += Int32(1)
                        end
                    end
                elseif x[idx,2] == x[idx,1]
                    if x[idx,4] > x[idx,3]
                        OUT[idx,1] = (log(max(x[idx,4], 0.0)) - log(max(x[idx,3], 0.0)))/(x[idx,4] - x[idx,3]) * (x[idx,1] - x[idx,3]) + log(max(x[idx,3], 0.0))
                        OUT[idx,2] = log(max(x[idx,1], 0.0))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * (log(max(x[idx,4], 0.0)) - log(max(x[idx,3], 0.0)))/(x[idx,4] - x[idx,3])
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (1/x[idx,1])
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = log(max(x[idx,3], 0.0))
                        OUT[idx,2] = log(max(x[idx,1], 0.0))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (1/x[idx,1])
                            col += Int32(1)
                        end
                    end
                elseif x[idx,3] >= x[idx,2]
                    if x[idx,4] > x[idx,3]
                        OUT[idx,1] = (log(max(x[idx,4], 0.0)) - log(max(x[idx,3], 0.0)))/(x[idx,4] - x[idx,3]) * (x[idx,2] - x[idx,3]) + log(max(x[idx,3], 0.0))
                        OUT[idx,2] = log(max(x[idx,1], 0.0))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * (log(max(x[idx,4], 0.0)) - log(max(x[idx,3], 0.0)))/(x[idx,4] - x[idx,3])
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (1/x[idx,1])
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = log(max(x[idx,3], 0.0))
                        OUT[idx,2] = log(max(x[idx,1], 0.0))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (1/x[idx,1])
                            col += Int32(1)
                        end
                    end
                else
                    OUT[idx,1] = log(max(x[idx,3], 0.0))
                    OUT[idx,2] = log(max(x[idx,1], 0.0))
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = 0.0
                        OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (1/x[idx,1])
                        col += Int32(1)
                    end
                end
            elseif x[idx,2] == x[idx,1]
                if x[idx,4] > x[idx,3]
                    OUT[idx,1] = (log(max(x[idx,4], 0.0)) - log(max(x[idx,3], 0.0)))/(x[idx,4] - x[idx,3]) * (x[idx,1] - x[idx,3]) + log(max(x[idx,3], 0.0))
                    OUT[idx,2] = log(max(x[idx,1], 0.0))
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * (log(max(x[idx,4], 0.0)) - log(max(x[idx,3], 0.0)))/(x[idx,4] - x[idx,3])
                        OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (1/x[idx,1])
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = log(max(x[idx,3], 0.0))
                    OUT[idx,2] = log(max(x[idx,1], 0.0))
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = 0.0
                        OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (1/x[idx,1])
                        col += Int32(1)
                    end
                end
            elseif x[idx,4] >= x[idx,2]
                if x[idx,1] >= x[idx,3]
                    if x[idx,4] > x[idx,3]
                        OUT[idx,1] = (log(max(x[idx,4], 0.0)) - log(max(x[idx,3], 0.0)))/(x[idx,4] - x[idx,3]) * (x[idx,1] - x[idx,3]) + log(max(x[idx,3], 0.0))
                        OUT[idx,2] = log(max(x[idx,2], 0.0))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * (log(max(x[idx,4], 0.0)) - log(max(x[idx,3], 0.0)))/(x[idx,4] - x[idx,3])
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (1/x[idx,2])
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = log(max(x[idx,3], 0.0))
                        OUT[idx,2] = log(max(x[idx,2], 0.0))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (1/x[idx,2])
                            col += Int32(1)
                        end
                    end
                elseif x[idx,3] >= x[idx,2]
                    if x[idx,4] > x[idx,3]
                        OUT[idx,1] = (log(max(x[idx,4], 0.0)) - log(max(x[idx,3], 0.0)))/(x[idx,4] - x[idx,3]) * (x[idx,2] - x[idx,3]) + log(max(x[idx,3], 0.0))
                        OUT[idx,2] = log(max(x[idx,2], 0.0))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * (log(max(x[idx,4], 0.0)) - log(max(x[idx,3], 0.0)))/(x[idx,4] - x[idx,3])
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (1/x[idx,2])
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = log(max(x[idx,3], 0.0))
                        OUT[idx,2] = log(max(x[idx,2], 0.0))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (1/x[idx,2])
                            col += Int32(1)
                        end
                    end
                else
                    OUT[idx,1] = log(max(x[idx,3], 0.0))
                    OUT[idx,2] = log(max(x[idx,2], 0.0))
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = 0.0
                        OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (1/x[idx,2])
                        col += Int32(1)
                    end
                end
            else
                if x[idx,1] >= x[idx,3]
                    if x[idx,4] > x[idx,3]
                        OUT[idx,1] = (log(max(x[idx,4], 0.0)) - log(max(x[idx,3], 0.0)))/(x[idx,4] - x[idx,3]) * (x[idx,1] - x[idx,3]) + log(max(x[idx,3], 0.0))
                        OUT[idx,2] = log(max(x[idx,4], 0.0))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * (log(max(x[idx,4], 0.0)) - log(max(x[idx,3], 0.0)))/(x[idx,4] - x[idx,3])
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = log(max(x[idx,3], 0.0))
                        OUT[idx,2] = log(max(x[idx,4], 0.0))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                elseif x[idx,3] >= x[idx,2]
                    if x[idx,4] > x[idx,3]
                        OUT[idx,1] = (log(max(x[idx,4], 0.0)) - log(max(x[idx,3], 0.0)))/(x[idx,4] - x[idx,3]) * (x[idx,2] - x[idx,3]) + log(max(x[idx,3], 0.0))
                        OUT[idx,2] = log(max(x[idx,4], 0.0))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * (log(max(x[idx,4], 0.0)) - log(max(x[idx,3], 0.0)))/(x[idx,4] - x[idx,3])
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = log(max(x[idx,3], 0.0))
                        OUT[idx,2] = log(max(x[idx,4], 0.0))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                else
                    OUT[idx,1] = log(max(x[idx,3], 0.0))
                    OUT[idx,2] = log(max(x[idx,4], 0.0))
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = 0.0
                        OUT[idx,end-1*colmax+col] = 0.0
                        col += Int32(1)
                    end
                end
            end
        elseif x[idx,4] >= x[idx,1]
            if x[idx,3] >= x[idx,1]
                if x[idx,4] > x[idx,3]
                    OUT[idx,1] = (log(max(x[idx,4], 0.0)) - log(max(x[idx,3], 0.0)))/(x[idx,4] - x[idx,3]) * (x[idx,1] - x[idx,3]) + log(max(x[idx,3], 0.0))
                    OUT[idx,2] = log(max(x[idx,1], 0.0))
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * (log(max(x[idx,4], 0.0)) - log(max(x[idx,3], 0.0)))/(x[idx,4] - x[idx,3])
                        OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (1/x[idx,1])
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = log(max(x[idx,3], 0.0))
                    OUT[idx,2] = log(max(x[idx,1], 0.0))
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = 0.0
                        OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (1/x[idx,1])
                        col += Int32(1)
                    end
                end
            elseif x[idx,2] >= x[idx,3]
                if x[idx,4] > x[idx,3]
                    OUT[idx,1] = (log(max(x[idx,4], 0.0)) - log(max(x[idx,3], 0.0)))/(x[idx,4] - x[idx,3]) * (x[idx,2] - x[idx,3]) + log(max(x[idx,3], 0.0))
                    OUT[idx,2] = log(max(x[idx,1], 0.0))
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * (log(max(x[idx,4], 0.0)) - log(max(x[idx,3], 0.0)))/(x[idx,4] - x[idx,3])
                        OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (1/x[idx,1])
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = log(max(x[idx,3], 0.0))
                    OUT[idx,2] = log(max(x[idx,1], 0.0))
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = 0.0
                        OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (1/x[idx,1])
                        col += Int32(1)
                    end
                end
            else
                OUT[idx,1] = log(max(x[idx,3], 0.0))
                OUT[idx,2] = log(max(x[idx,1], 0.0))
                while col <= colmax
                    OUT[idx,end-2*colmax+col] = 0.0
                    OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (1/x[idx,1])
                    col += Int32(1)
                end
            end
        elseif x[idx,2] >= x[idx,4]
            if x[idx,3] >= x[idx,1]
                if x[idx,4] > x[idx,3]
                    OUT[idx,1] = (log(max(x[idx,4], 0.0)) - log(max(x[idx,3], 0.0)))/(x[idx,4] - x[idx,3]) * (x[idx,1] - x[idx,3]) + log(max(x[idx,3], 0.0))
                    OUT[idx,2] = log(max(x[idx,2], 0.0))
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * (log(max(x[idx,4], 0.0)) - log(max(x[idx,3], 0.0)))/(x[idx,4] - x[idx,3])
                        OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (1/x[idx,2])
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = log(max(x[idx,3], 0.0))
                    OUT[idx,2] = log(max(x[idx,2], 0.0))
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = 0.0
                        OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (1/x[idx,2])
                        col += Int32(1)
                    end
                end
            elseif x[idx,2] >= x[idx,3]
                if x[idx,4] > x[idx,3]
                    OUT[idx,1] = (log(max(x[idx,4], 0.0)) - log(max(x[idx,3], 0.0)))/(x[idx,4] - x[idx,3]) * (x[idx,2] - x[idx,3]) + log(max(x[idx,3], 0.0))
                    OUT[idx,2] = log(max(x[idx,2], 0.0))
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * (log(max(x[idx,4], 0.0)) - log(max(x[idx,3], 0.0)))/(x[idx,4] - x[idx,3])
                        OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (1/x[idx,2])
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = log(max(x[idx,3], 0.0))
                    OUT[idx,2] = log(max(x[idx,2], 0.0))
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = 0.0
                        OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (1/x[idx,2])
                        col += Int32(1)
                    end
                end
            else
                OUT[idx,1] = log(max(x[idx,3], 0.0))
                OUT[idx,2] = log(max(x[idx,2], 0.0))
                while col <= colmax
                    OUT[idx,end-2*colmax+col] = 0.0
                    OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (1/x[idx,2])
                    col += Int32(1)
                end
            end
        else
            if x[idx,3] >= x[idx,1]
                if x[idx,4] > x[idx,3]
                    OUT[idx,1] = (log(max(x[idx,4], 0.0)) - log(max(x[idx,3], 0.0)))/(x[idx,4] - x[idx,3]) * (x[idx,1] - x[idx,3]) + log(max(x[idx,3], 0.0))
                    OUT[idx,2] = log(max(x[idx,4], 0.0))
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * (log(max(x[idx,4], 0.0)) - log(max(x[idx,3], 0.0)))/(x[idx,4] - x[idx,3])
                        OUT[idx,end-1*colmax+col] = 0.0
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = log(max(x[idx,3], 0.0))
                    OUT[idx,2] = log(max(x[idx,4], 0.0))
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = 0.0
                        OUT[idx,end-1*colmax+col] = 0.0
                        col += Int32(1)
                    end
                end
            elseif x[idx,2] >= x[idx,3]
                if x[idx,4] > x[idx,3]
                    OUT[idx,1] = (log(max(x[idx,4], 0.0)) - log(max(x[idx,3], 0.0)))/(x[idx,4] - x[idx,3]) * (x[idx,2] - x[idx,3]) + log(max(x[idx,3], 0.0))
                    OUT[idx,2] = log(max(x[idx,4], 0.0))
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * (log(max(x[idx,4], 0.0)) - log(max(x[idx,3], 0.0)))/(x[idx,4] - x[idx,3])
                        OUT[idx,end-1*colmax+col] = 0.0
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = log(max(x[idx,3], 0.0))
                    OUT[idx,2] = log(max(x[idx,4], 0.0))
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = 0.0
                        OUT[idx,end-1*colmax+col] = 0.0
                        col += Int32(1)
                    end
                end
            else
                OUT[idx,1] = log(max(x[idx,3], 0.0))
                OUT[idx,2] = log(max(x[idx,4], 0.0))
                while col <= colmax
                    OUT[idx,end-2*colmax+col] = 0.0
                    OUT[idx,end-1*colmax+col] = 0.0
                    col += Int32(1)
                end
            end
        end

        # Convert domain errors to NaNs
        if x[idx,3] < 0.0
            OUT[idx,1] = NaN
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-2*colmax+col] = NaN
                col += Int32(1)
            end
        end
        if x[idx,1] < 0.0
            OUT[idx,1] = NaN
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-2*colmax+col] = NaN
                col += Int32(1)
            end
        end
        if x[idx,2] < 0.0
            OUT[idx,2] = NaN
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-1*colmax+col] = NaN
                col += Int32(1)
            end
        end
        if x[idx,4] < 0.0
            OUT[idx,2] = NaN
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-1*colmax+col] = NaN
                col += Int32(1)
            end
        end

        # Perform the cut operation
        if OUT[idx,1] < OUT[idx,3]
            OUT[idx,1] = OUT[idx,3]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-2*colmax+col] = 0.0
                col += Int32(1)
            end
        end
        if OUT[idx,2] > OUT[idx,4]
            OUT[idx,2] = OUT[idx,4]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-1*colmax+col] = 0.0
                col += Int32(1)
            end
        end

        idx += stride
    end
    return nothing
end

# Inversion
# max threads: 768
function SCMC_inv_kernel(OUT::CuDeviceMatrix, x::CuDeviceMatrix)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    colmax = Int32((size(OUT,2)-4)/2)

    while idx <= Int32(size(OUT,1))
        # Reset the column counter
        col = Int32(1)

        # Begin inversion
        if x[idx,3] > 0.0 # inv1(x)
            OUT[idx,3] = inv(x[idx,4])
            OUT[idx,4] = inv(x[idx,3])
            if x[idx,2] >= x[idx,1]
                if x[idx,1] >= x[idx,4]
                    if x[idx,1] >= x[idx,3]
                        OUT[idx,1] = (1.0 / x[idx,1])
                        OUT[idx,2] = (x[idx,4] + x[idx,3] - x[idx,1])/(x[idx,3]*x[idx,4])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = -1.0/(x[idx,1]*x[idx,1]) * x[idx,end-2*colmax+col]
                            OUT[idx,end-1*colmax+col] = -1.0/(x[idx,3]*x[idx,4]) * x[idx,end-2*colmax+col]
                            col += Int32(1)
                        end
                    elseif x[idx,2] == x[idx,1]
                        OUT[idx,1] = (1.0 / x[idx,1])
                        OUT[idx,2] = (x[idx,4] + x[idx,3] - x[idx,1])/(x[idx,3]*x[idx,4])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = -1.0/(x[idx,1]*x[idx,1]) * x[idx,end-2*colmax+col]
                            OUT[idx,end-1*colmax+col] = -1.0/(x[idx,3]*x[idx,4]) * x[idx,end-2*colmax+col]
                            col += Int32(1)
                        end
                    elseif x[idx,3] >= x[idx,2]
                        OUT[idx,1] = (1.0 / x[idx,1])
                        OUT[idx,2] = (x[idx,4] + x[idx,3] - x[idx,2])/(x[idx,3]*x[idx,4])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = -1.0/(x[idx,1]*x[idx,1]) * x[idx,end-2*colmax+col]
                            OUT[idx,end-1*colmax+col] = -1.0/(x[idx,3]*x[idx,4]) * x[idx,end-1*colmax+col]
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = (1.0 / x[idx,1])
                        OUT[idx,2] = (1.0 / x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = -1.0/(x[idx,1]*x[idx,1]) * x[idx,end-2*colmax+col]
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                elseif x[idx,2] == x[idx,1]
                    OUT[idx,1] = (1.0 / x[idx,1])
                    OUT[idx,2] = (x[idx,4] + x[idx,3] - x[idx,1])/(x[idx,3]*x[idx,4])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = -1.0/(x[idx,1]*x[idx,1]) * x[idx,end-2*colmax+col]
                        OUT[idx,end-1*colmax+col] = -1.0/(x[idx,3]*x[idx,4]) * x[idx,end-2*colmax+col]
                        col += Int32(1)
                    end
                elseif x[idx,4] >= x[idx,2]
                    if x[idx,1] >= x[idx,3]
                        OUT[idx,1] = (1.0 / x[idx,2])
                        OUT[idx,2] = (x[idx,4] + x[idx,3] - x[idx,1])/(x[idx,3]*x[idx,4])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = -1.0/(x[idx,2]*x[idx,2]) * x[idx,end-1*colmax+col]
                            OUT[idx,end-1*colmax+col] = -1.0/(x[idx,3]*x[idx,4]) * x[idx,end-2*colmax+col]
                            col += Int32(1)
                        end
                    elseif x[idx,3] >= x[idx,2]
                        OUT[idx,1] = (1.0 / x[idx,2])
                        OUT[idx,2] = (x[idx,4] + x[idx,3] - x[idx,2])/(x[idx,3]*x[idx,4])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = -1.0/(x[idx,2]*x[idx,2]) * x[idx,end-1*colmax+col]
                            OUT[idx,end-1*colmax+col] = -1.0/(x[idx,3]*x[idx,4]) * x[idx,end-1*colmax+col]
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = (1.0 / x[idx,2])
                        OUT[idx,2] = (1.0 / x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = -1.0/(x[idx,2]*x[idx,2]) * x[idx,end-1*colmax+col]
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                else
                    if x[idx,1] >= x[idx,3]
                        OUT[idx,1] = (1.0 / x[idx,4])
                        OUT[idx,2] = (x[idx,4] + x[idx,3] - x[idx,1])/(x[idx,3]*x[idx,4])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = -1.0/(x[idx,3]*x[idx,4]) * x[idx,end-2*colmax+col]
                            col += Int32(1)
                        end
                    elseif x[idx,3] >= x[idx,2]
                        OUT[idx,1] = (1.0 / x[idx,4])
                        OUT[idx,2] = (x[idx,4] + x[idx,3] - x[idx,2])/(x[idx,3]*x[idx,4])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = -1.0/(x[idx,3]*x[idx,4]) * x[idx,end-1*colmax+col]
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = (1.0 / x[idx,4])
                        OUT[idx,2] = (1.0 / x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                end
            elseif x[idx,4] >= x[idx,1]
                if x[idx,3] >= x[idx,1]
                    OUT[idx,1] = (1.0 / x[idx,1])
                    OUT[idx,2] = (x[idx,4] + x[idx,3] - x[idx,1])/(x[idx,3]*x[idx,4])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = -1.0/(x[idx,1]*x[idx,1]) * x[idx,end-2*colmax+col]
                        OUT[idx,end-1*colmax+col] = -1.0/(x[idx,3]*x[idx,4]) * x[idx,end-2*colmax+col]
                        col += Int32(1)
                    end
                elseif x[idx,2] >= x[idx,3]
                    OUT[idx,1] = (1.0 / x[idx,1])
                    OUT[idx,2] = (x[idx,4] + x[idx,3] - x[idx,2])/(x[idx,3]*x[idx,4])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = -1.0/(x[idx,1]*x[idx,1]) * x[idx,end-2*colmax+col]
                        OUT[idx,end-1*colmax+col] = -1.0/(x[idx,3]*x[idx,4]) * x[idx,end-1*colmax+col]
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = (1.0 / x[idx,1])
                    OUT[idx,2] = (1.0 / x[idx,3])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = -1.0/(x[idx,1]*x[idx,1]) * x[idx,end-2*colmax+col]
                        OUT[idx,end-1*colmax+col] = 0.0
                        col += Int32(1)
                    end
                end
            elseif x[idx,2] >= x[idx,4]
                if x[idx,3] >= x[idx,1]
                    OUT[idx,1] = (1.0 / x[idx,2])
                    OUT[idx,2] = (x[idx,4] + x[idx,3] - x[idx,1])/(x[idx,3]*x[idx,4])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = -1.0/(x[idx,2]*x[idx,2]) * x[idx,end-1*colmax+col]
                        OUT[idx,end-1*colmax+col] = -1.0/(x[idx,3]*x[idx,4]) * x[idx,end-2*colmax+col]
                        col += Int32(1)
                    end
                elseif x[idx,2] >= x[idx,3]
                    OUT[idx,1] = (1.0 / x[idx,2])
                    OUT[idx,2] = (x[idx,4] + x[idx,3] - x[idx,2])/(x[idx,3]*x[idx,4])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = -1.0/(x[idx,2]*x[idx,2]) * x[idx,end-1*colmax+col]
                        OUT[idx,end-1*colmax+col] = -1.0/(x[idx,3]*x[idx,4]) * x[idx,end-1*colmax+col]
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = (1.0 / x[idx,2])
                    OUT[idx,2] = (1.0 / x[idx,3])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = -1.0/(x[idx,2]*x[idx,2]) * x[idx,end-1*colmax+col]
                        OUT[idx,end-1*colmax+col] = 0.0
                        col += Int32(1)
                    end
                end
            else
                if x[idx,3] >= x[idx,1]
                    OUT[idx,1] = (1.0 / x[idx,4])
                    OUT[idx,2] = (x[idx,4] + x[idx,3] - x[idx,1])/(x[idx,3]*x[idx,4])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = 0.0
                        OUT[idx,end-1*colmax+col] = -1.0/(x[idx,3]*x[idx,4]) * x[idx,end-2*colmax+col]
                        col += Int32(1)
                    end
                elseif x[idx,2] >= x[idx,3]
                    OUT[idx,1] = (1.0 / x[idx,4])
                    OUT[idx,2] = (x[idx,4] + x[idx,3] - x[idx,2])/(x[idx,3]*x[idx,4])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = 0.0
                        OUT[idx,end-1*colmax+col] = -1.0/(x[idx,3]*x[idx,4]) * x[idx,end-1*colmax+col]
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = (1.0 / x[idx,4])
                    OUT[idx,2] = (1.0 / x[idx,3])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = 0.0
                        OUT[idx,end-1*colmax+col] = 0.0
                        col += Int32(1)
                    end
                end
            end
        elseif x[idx,4] < 0.0 # -inv(-x)
            OUT[idx,3] = -inv(-x[idx,4])
            OUT[idx,4] = -inv(-x[idx,3])
            if x[idx,1] <= x[idx,2]
                if x[idx,2]== x[idx,1]
                    OUT[idx,1] = (x[idx,3] + x[idx,4] - x[idx,2])/(x[idx,3]*x[idx,4])
                    OUT[idx,2] = 1.0/x[idx,2]
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = (-1.0/(x[idx,3]*x[idx,4])) * x[idx,end-1*colmax+col]
                        OUT[idx,end-1*colmax+col] = (-1.0/(x[idx,2]*x[idx,2])) * x[idx,end-1*colmax+col]
                        col += Int32(1)
                    end
                elseif x[idx,2] <= x[idx,4]
                    if x[idx,2] <= x[idx,3]
                        OUT[idx,1] = (x[idx,3] + x[idx,4] - x[idx,2])/(x[idx,3]*x[idx,4])
                        OUT[idx,2] = 1.0/x[idx,2]
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = (-1.0/(x[idx,3]*x[idx,4])) * x[idx,end-1*colmax+col]
                            OUT[idx,end-1*colmax+col] = (-1.0/(x[idx,2]*x[idx,2])) * x[idx,end-1*colmax+col]
                            col += Int32(1)
                        end
                    elseif x[idx,3] <= x[idx,1]
                        OUT[idx,1] = (x[idx,3] + x[idx,4] - x[idx,2])/(x[idx,3]*x[idx,4])
                        OUT[idx,2] = 1.0/x[idx,1]
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = (-1.0/(x[idx,3]*x[idx,4])) * x[idx,end-1*colmax+col]
                            OUT[idx,end-1*colmax+col] = (-1.0/(x[idx,1]*x[idx,1])) * x[idx,end-2*colmax+col]
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = (x[idx,3] + x[idx,4] - x[idx,2])/(x[idx,3]*x[idx,4])
                        OUT[idx,2] = 1.0/x[idx,3] 
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = (-1.0/(x[idx,3]*x[idx,4])) * x[idx,end-1*colmax+col]
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                elseif x[idx,4] <= x[idx,1]
                    if x[idx,2] <= x[idx,3]
                        OUT[idx,1] = (x[idx,3] + x[idx,4] - x[idx,1])/(x[idx,3]*x[idx,4])
                        OUT[idx,2] = 1.0/x[idx,2]
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = (-1.0/(x[idx,3]*x[idx,4])) * x[idx,end-2*colmax+col]
                            OUT[idx,end-1*colmax+col] = (-1.0/(x[idx,2]*x[idx,2])) * x[idx,end-1*colmax+col]
                            col += Int32(1)
                        end
                    elseif x[idx,3] <= x[idx,1]
                        OUT[idx,1] = (x[idx,3] + x[idx,4] - x[idx,1])/(x[idx,3]*x[idx,4])
                        OUT[idx,2] = 1.0/x[idx,1]
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = (-1.0/(x[idx,3]*x[idx,4])) * x[idx,end-2*colmax+col]
                            OUT[idx,end-1*colmax+col] = (-1.0/(x[idx,1]*x[idx,1])) * x[idx,end-2*colmax+col]
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = (x[idx,3] + x[idx,4] - x[idx,1])/(x[idx,3]*x[idx,4])
                        OUT[idx,2] = 1.0/x[idx,3] 
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = (-1.0/(x[idx,3]*x[idx,4])) * x[idx,end-2*colmax+col]
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                else
                    if x[idx,2] <= x[idx,3]
                        OUT[idx,1] = 1.0/x[idx,4]
                        OUT[idx,2] = 1.0/x[idx,2]
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = (-1.0/(x[idx,2]*x[idx,2])) * x[idx,end-1*colmax+col]
                            col += Int32(1)
                        end
                    elseif x[idx,3] <= x[idx,1]
                        OUT[idx,1] = 1.0/x[idx,4]
                        OUT[idx,2] = 1.0/x[idx,1]
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = (-1.0/(x[idx,1]*x[idx,1])) * x[idx,end-2*colmax+col]
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = 1.0/x[idx,4]
                        OUT[idx,2] = 1.0/x[idx,3]
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                end
            elseif x[idx,4] <= x[idx,2]
                if x[idx,3] <= x[idx,2]
                    OUT[idx,1] = (x[idx,3] + x[idx,4] - x[idx,2])/(x[idx,3]*x[idx,4])
                    OUT[idx,2] = 1.0/x[idx,2]
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = (-1.0/(x[idx,3]*x[idx,4])) * x[idx,end-1*colmax+col]
                        OUT[idx,end-1*colmax+col] = (-1.0/(x[idx,2]*x[idx,2])) * x[idx,end-1*colmax+col]
                        col += Int32(1)
                    end
                elseif x[idx,1] <= x[idx,3]
                    OUT[idx,1] = (x[idx,3] + x[idx,4] - x[idx,2])/(x[idx,3]*x[idx,4])
                    OUT[idx,2] = 1.0/x[idx,1]
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = (-1.0/(x[idx,3]*x[idx,4])) * x[idx,end-1*colmax+col]
                        OUT[idx,end-1*colmax+col] = (-1.0/(x[idx,1]*x[idx,1])) * x[idx,end-2*colmax+col]
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = (x[idx,3] + x[idx,4] - x[idx,2])/(x[idx,3]*x[idx,4])
                    OUT[idx,2] = 1.0/x[idx,3] 
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = (-1.0/(x[idx,3]*x[idx,4])) * x[idx,end-1*colmax+col]
                        OUT[idx,end-1*colmax+col] = 0.0
                        col += Int32(1)
                    end
                end
            elseif x[idx,1] <= x[idx,4]
                if x[idx,3] <= x[idx,2]
                    OUT[idx,1] = (x[idx,3] + x[idx,4] - x[idx,1])/(x[idx,3]*x[idx,4])
                    OUT[idx,2] = 1.0/x[idx,2]
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = (-1.0/(x[idx,3]*x[idx,4])) * x[idx,end-2*colmax+col]
                        OUT[idx,end-1*colmax+col] = (-1.0/(x[idx,2]*x[idx,2])) * x[idx,end-1*colmax+col]
                        col += Int32(1)
                    end
                elseif x[idx,1] <= x[idx,3]
                    OUT[idx,1] = (x[idx,3] + x[idx,4] - x[idx,1])/(x[idx,3]*x[idx,4])
                    OUT[idx,2] = 1.0/x[idx,1]
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = (-1.0/(x[idx,3]*x[idx,4])) * x[idx,end-2*colmax+col]
                        OUT[idx,end-1*colmax+col] = (-1.0/(x[idx,1]*x[idx,1])) * x[idx,end-2*colmax+col]
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = (x[idx,3] + x[idx,4] - x[idx,1])/(x[idx,3]*x[idx,4])
                    OUT[idx,2] = 1.0/x[idx,3] 
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = (-1.0/(x[idx,3]*x[idx,4])) * x[idx,end-2*colmax+col]
                        OUT[idx,end-1*colmax+col] = 0.0
                        col += Int32(1)
                    end
                end
            else
                if x[idx,3] <= x[idx,2]
                    OUT[idx,1] = 1.0/x[idx,4]
                    OUT[idx,2] = 1.0/x[idx,2]
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = 0.0
                        OUT[idx,end-1*colmax+col] = (-1.0/(x[idx,2]*x[idx,2])) * x[idx,end-1*colmax+col]
                        col += Int32(1)
                    end
                elseif x[idx,1] <= x[idx,3]
                    OUT[idx,1] = 1.0/x[idx,4]
                    OUT[idx,2] = 1.0/x[idx,1]
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = 0.0
                        OUT[idx,end-1*colmax+col] = (-1.0/(x[idx,1]*x[idx,1])) * x[idx,end-2*colmax+col]
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = 1.0/x[idx,4]
                    OUT[idx,2] = 1.0/x[idx,3]
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = 0.0
                        OUT[idx,end-1*colmax+col] = 0.0
                        col += Int32(1)
                    end
                end
            end
        else #invalid
            OUT[idx,3] = NaN
            OUT[idx,4] = NaN
            OUT[idx,1] = NaN
            OUT[idx,2] = NaN
            while col <= colmax
                OUT[idx,end-2*colmax+col] = NaN
                OUT[idx,end-1*colmax+col] = NaN
                col += Int32(1)
            end
        end

        
        # Perform the cut operation
        if OUT[idx,1] < OUT[idx,3]
            OUT[idx,1] = OUT[idx,3]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-2*colmax+col] = 0.0
                col += Int32(1)
            end
        end
        if OUT[idx,2] > OUT[idx,4]
            OUT[idx,2] = OUT[idx,4]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-1*colmax+col] = 0.0
                col += Int32(1)
            end
        end

        idx += stride
    end
    return nothing
end

# Multiplication by a constant
# max threads: 640
function SCMC_cmul_kernel(OUT::CuDeviceMatrix, CONST::Real, x::CuDeviceMatrix)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    colmax = Int32((size(OUT,2)-4)/2)

    while idx <= Int32(size(OUT,1))
        # Reset the column counter
        col = Int32(1)

        # Begin multiplication by a constant
        if CONST >= 0.0
            OUT[idx,3] = CONST*x[idx,3]
            OUT[idx,4] = CONST*x[idx,4]
            OUT[idx,1] = CONST*x[idx,1]
            OUT[idx,2] = CONST*x[idx,2]
            while col <= colmax
                OUT[idx,end-2*colmax+col] = CONST*x[idx,end-2*colmax+col]
                OUT[idx,end-1*colmax+col] = CONST*x[idx,end-1*colmax+col]
                col += Int32(1)
            end
        else
            OUT[idx,3] = CONST*x[idx,4]
            OUT[idx,4] = CONST*x[idx,3]
            OUT[idx,1] = CONST*x[idx,2]
            OUT[idx,2] = CONST*x[idx,1]
            while col <= colmax
                OUT[idx,end-2*colmax+col] = CONST*x[idx,end-1*colmax+col]
                OUT[idx,end-1*colmax+col] = CONST*x[idx,end-2*colmax+col]
                col += Int32(1)
            end
        end

        # Perform the cut operation
        if OUT[idx,1] < OUT[idx,3]
            OUT[idx,1] = OUT[idx,3]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-2*colmax+col] = 0.0
                col += Int32(1)
            end
        end
        if OUT[idx,2] > OUT[idx,4]
            OUT[idx,2] = OUT[idx,4]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-1*colmax+col] = 0.0
                col += Int32(1)
            end
        end

        idx += stride
    end
    return nothing
end

# Sigmoid function
# max threads: 640
function SCMC_sigmoid_kernel(OUT::CuDeviceMatrix, x::CuDeviceMatrix)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    colmax = Int32((size(OUT,2)-4)/2)

    while idx <= Int32(size(OUT,1))
        # Reset the column counter
        col = Int32(1)

        # Interval is independent of the rest
        OUT[idx,3] = 1.0/(1.0+exp(-x[idx,3]))
        OUT[idx,4] = 1.0/(1.0+exp(-x[idx,4]))


        if x[idx,3] >= 0.0
            if x[idx,2] >= x[idx,1]
                if x[idx,1] >= x[idx,4]
                    if x[idx,1] >= x[idx,3]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,1]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))/(x[idx,4] - x[idx,3])
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                                col += Int32(1)
                            end
                        end
                    elseif x[idx,2] == x[idx,1]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,1]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))/(x[idx,4] - x[idx,3])
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                                col += Int32(1)
                            end
                        end
                    elseif x[idx,3] >= x[idx,2]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,2]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))/(x[idx,4] - x[idx,3])
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                                col += Int32(1)
                            end
                        end
                    else
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,3]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,3] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                                col += Int32(1)
                            end
                        end
                    end
                elseif x[idx,2] == x[idx,1]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,1]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))/(x[idx,4] - x[idx,3])
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            col += Int32(1)
                        end
                    end
                elseif x[idx,4] >= x[idx,2]
                    if x[idx,1] >= x[idx,3]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,1]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))/(x[idx,4] - x[idx,3])
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                                col += Int32(1)
                            end
                        end
                    elseif x[idx,3] >= x[idx,2]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,2]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))/(x[idx,4] - x[idx,3])
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                                col += Int32(1)
                            end
                        end
                    else
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,3]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,3] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                                col += Int32(1)
                            end
                        end
                    end
                else
                    if x[idx,1] >= x[idx,3]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,1]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))/(x[idx,4] - x[idx,3])
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        end
                    elseif x[idx,3] >= x[idx,2]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,2]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))/(x[idx,4] - x[idx,3])
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        end
                    else
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,3]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,3] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        end
                    end
                end
            elseif x[idx,4] >= x[idx,1]
                if x[idx,3] >= x[idx,1]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,1]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))/(x[idx,4] - x[idx,3])
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            col += Int32(1)
                        end
                    end
                elseif x[idx,2] >= x[idx,3]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,2]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))/(x[idx,4] - x[idx,3])
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            col += Int32(1)
                        end
                    end
                else
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,3]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,3] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            col += Int32(1)
                        end
                    end
                end
            elseif x[idx,2] >= x[idx,4]
                if x[idx,3] >= x[idx,1]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,1]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))/(x[idx,4] - x[idx,3])
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                            col += Int32(1)
                        end
                    end
                elseif x[idx,2] >= x[idx,3]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,2]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))/(x[idx,4] - x[idx,3])
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                            col += Int32(1)
                        end
                    end
                else
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,3]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,3] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                            col += Int32(1)
                        end
                    end
                end
            else
                if x[idx,3] >= x[idx,1]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,1]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))/(x[idx,4] - x[idx,3])
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                elseif x[idx,2] >= x[idx,3]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,2]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))/(x[idx,4] - x[idx,3])
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                else
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,3]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,3] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                end
            end
        elseif x[idx,4] <= 0.0
            if x[idx,2] >= x[idx,1]
                if x[idx,1] >= x[idx,4]
                    if x[idx,1] >= x[idx,3]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                            OUT[idx,2] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,1]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    elseif x[idx,2] == x[idx,1]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                            OUT[idx,2] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,1]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    elseif x[idx,3] >= x[idx,2]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                            OUT[idx,2] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,1]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    else
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                            OUT[idx,2] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,1]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    end
                elseif x[idx,2] == x[idx,1]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                        OUT[idx,2] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,1]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                elseif x[idx,4] >= x[idx,2]
                    if x[idx,1] >= x[idx,3]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                            OUT[idx,2] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,2]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    elseif x[idx,3] >= x[idx,2]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                            OUT[idx,2] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,2]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    else
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                            OUT[idx,2] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,2]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    end
                else
                    if x[idx,1] >= x[idx,3]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                            OUT[idx,2] = ((1.0/(1.0 + exp(-x[idx,4])))*(x[idx,4] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        end
                    elseif x[idx,3] >= x[idx,2]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                            OUT[idx,2] = ((1.0/(1.0 + exp(-x[idx,4])))*(x[idx,4] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        end
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                            OUT[idx,2] = ((1.0/(1.0 + exp(-x[idx,4])))*(x[idx,4] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        end
                    else
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                            OUT[idx,2] = ((1.0/(1.0 + exp(-x[idx,4])))*(x[idx,4] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        end
                    end
                end
            elseif x[idx,4] >= x[idx,1]
                if x[idx,3] >= x[idx,1]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                        OUT[idx,2] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,1]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                elseif x[idx,2] >= x[idx,3]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                        OUT[idx,2] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,1]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                else
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                        OUT[idx,2] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,1]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                end
            elseif x[idx,2] >= x[idx,4]
                if x[idx,3] >= x[idx,1]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                        OUT[idx,2] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,2]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                elseif x[idx,2] >= x[idx,3]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                        OUT[idx,2] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,2]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                else
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                        OUT[idx,2] = ((1.0/(1.0 + exp(-x[idx,3])))*(x[idx,4] - x[idx,2]) + (1.0/(1.0 + exp(-x[idx,4])))*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                end
            else
                if x[idx,3] >= x[idx,1]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                        OUT[idx,2] = ((1.0/(1.0 + exp(-x[idx,4])))*(x[idx,4] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                elseif x[idx,2] >= x[idx,3]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                        OUT[idx,2] = ((1.0/(1.0 + exp(-x[idx,4])))*(x[idx,4] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                else
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                        OUT[idx,2] = ((1.0/(1.0 + exp(-x[idx,4])))*(x[idx,4] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                end
            end
        else
            # This is the hard one.

            # Do the whole cv/cc thing to get CV (midcv) and CC (midcc)

            # cv, dcv, cv_p = cv_sigmoid(midcv, xL, xU, cv_p)
            # cc, dcc, cc_p = cc_sigmoid(midcc, xL, xU, cc_p)
            # cvgrad = midgrad(ccgrad, cvgrad, cv_id)*dcv
            # ccgrad = midgrad(ccgrad, cvgrad, cc_id)*dcc

            # FOR CV
            xkm = x[idx,3] # max(x_lo, min(x_hi, x_lo)). x_lo<x_hi by definition, so max(x_lo, x_lo)
            xk_cv = 0.0 # max(x_lo, min(x_hi, 0.0)). x_hi is positive and x_lo is negative in this condition, so 0.0
            fkm = (xkm - x[idx,4])*exp(-xkm)/(1.0 + exp(-xkm))^2 - (1.0/(1.0 + exp(-xkm)) - 1.0/(1.0 + exp(-x[idx,4])))
            flag = true
            iter = Int32(1)
            while iter <= Int32(100)
                fk = (xk_cv - x[idx,4])*exp(-xk_cv)/(1.0 + exp(-xk_cv))^2 - (1.0/(1.0 + exp(-xk_cv)) - 1.0/(1.0 + exp(-x[idx,4])))
                Bk = (fk - fkm)/(xk_cv - xkm)
                if (abs(fk) < 1.0e-10)
                    flag = false
                    break # use xk_cv
                elseif (Bk == 0.0) 
                    xk_cv = 0.0
                    break # Need to do golden section
                elseif (xk_cv == x[idx,3]) && (fk/Bk > 0.0)
                    flag = false
                    break # use xk_cv
                elseif (xk_cv == 0.0) && (fk/Bk < 0.0)
                    flag = false
                    break # use xk_cv
                end
                xkm = xk_cv
                fkm = fk
                xk_cv = max(x[idx,3], min(0.0, xk_cv - fk/Bk))
                iter += Int32(1)
            end

            # If flag, we need to do golden section instead.
            if flag
                a_sigmoid = x[idx,3]
                fa_sigmoid = (x[idx,3] - x[idx,4])*exp(-x[idx,3])/(1.0 + exp(-x[idx,3]))^2 - (1.0/(1.0 + exp(-x[idx,3])) - 1.0/(1.0 + exp(-x[idx,4])))
                c_sigmoid = 0.0
                fc_sigmoid = (0.0 - x[idx,4])*exp(0.0)/(1.0 + exp(0.0))^2 - (1.0/(1.0 + exp(0.0)) - 1.0/(1.0 + exp(-x[idx,4])))

                if fa_sigmoid*fc_sigmoid > 0.0
                    xk_cv = NaN
                end
                b_sigmoid = 0.0 - (2.0 - Base.MathConstants.golden)*(0.0 - x[idx,3])
                fb_sigmoid = (b_sigmoid - x[idx,4])*exp(-b_sigmoid)/(1.0 + exp(-b_sigmoid))^2 - (1.0/(1.0 + exp(-b_sigmoid)) - 1.0/(1.0 + exp(-x[idx,4])))

                iter = Int32(1)
                while iter <= Int32(100)
                    if (c_sigmoid - b_sigmoid > b_sigmoid - a_sigmoid)
                        x_sigmoid = b_sigmoid + (2.0 - Base.MathConstants.golden)*(c_sigmoid - b_sigmoid)
                        if abs(c_sigmoid-a_sigmoid) < 1.0e-10*(abs(b_sigmoid) + abs(x_sigmoid)) || iter == Int32(100)
                            xk_cv = (c_sigmoid + a_sigmoid)/2.0
                            break
                        end
                        iter += Int32(1)
                        fx_sigmoid = (x_sigmoid - x[idx,4])*exp(-x_sigmoid)/(1.0 + exp(-x_sigmoid))^2 - (1.0/(1.0 + exp(-x_sigmoid)) - 1.0/(1.0 + exp(-x[idx,4])))
                        if fa_sigmoid*fx_sigmoid < 0.0
                            c_sigmoid = x_sigmoid
                            fc_sigmoid = fx_sigmoid
                        else
                            a_sigmoid = b_sigmoid
                            fa_sigmoid = fb_sigmoid
                            b_sigmoid = x_sigmoid
                            fb_sigmoid = fx_sigmoid
                        end
                    else
                        x_sigmoid = b_sigmoid - (2.0 - Base.MathConstants.golden)*(b_sigmoid - a_sigmoid)
                        if abs(c_sigmoid-a_sigmoid) < 1.0e-10*(abs(b_sigmoid) + abs(x_sigmoid)) || iter == Int32(100)
                            xk_cv = (c_sigmoid + a_sigmoid)/2.0
                            break
                        end
                        iter += Int32(1)
                        fx_sigmoid = (x_sigmoid - x[idx,4])*exp(-x_sigmoid)/(1.0 + exp(-x_sigmoid))^2 - (1.0/(1.0 + exp(-x_sigmoid)) - 1.0/(1.0 + exp(-x[idx,4])))
                        if fa_sigmoid*fb_sigmoid < 0.0
                            c_sigmoid = b_sigmoid
                            fc_sigmoid = fb_sigmoid
                            b_sigmoid = x_sigmoid
                            fb_sigmoid = fx_sigmoid
                        else
                            a_sigmoid = x_sigmoid
                            fa_sigmoid = fx_sigmoid
                        end
                    end
                end
            end

            # FOR CC
            xkm = 0.0 # max(x_lo, min(x_hi, 0.0)). x_hi is positive in this condition, and x_lo is negative. So 0.0.
            xk_cc = x[idx,4] # max(x_lo, min(x_hi, x_hi)) == max(x_lo, x_hi), and x_hi>x_lo by definition.
            fkm = (xkm - x[idx,3])*exp(-xkm)/(1.0 + exp(-xkm))^2 - (1.0/(1.0 + exp(-xkm)) - 1.0/(1.0 + exp(-x[idx,3])))
            flag = true
            iter = Int32(1)
            while iter <= Int32(100)
                fk = (xk_cc - x[idx,3])*exp(-xk_cc)/(1.0 + exp(-xk_cc))^2 - (1.0/(1.0 + exp(-xk_cc)) - 1.0/(1.0 + exp(-x[idx,3])))
                Bk = (fk - fkm)/(xk_cc - xkm)
                if (abs(fk) < 1.0e-10)
                    flag = false
                    break # use xk_cc
                elseif (Bk == 0.0) 
                    xk_cc = 0.0
                    break # Need to do golden section
                elseif (xk_cc == 0.0) && (fk/Bk > 0.0)
                    flag = false
                    break # use xk_cc
                elseif (xk_cc == x[idx,4]) && (fk/Bk < 0.0)
                    flag = false
                    break # use xk_cc
                end
                xkm = xk_cc
                fkm = fk
                xk_cc = max(0.0, min(x[idx,4], xk_cc - fk/Bk))
                iter += Int32(1)
            end

            # If flag, we need to do golden section instead.
            if flag
                a_sigmoid = 0.0
                fa_sigmoid = (0.0 - x[idx,3])*exp(0.0)/(1.0 + exp(0.0))^2 - (1.0/(1.0 + exp(0.0)) - 1.0/(1.0 + exp(-x[idx,3])))
                c_sigmoid = x[idx,4]
                fc_sigmoid = (x[idx,4] - x[idx,3])*exp(-x[idx,4])/(1.0 + exp(-x[idx,4]))^2 - (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-x[idx,3])))

                if fa_sigmoid*fc_sigmoid > 0.0
                    xk_cc = NaN
                end

                b_sigmoid = x[idx,4] - (2.0 - Base.MathConstants.golden)*(x[idx,4] - 0.0)
                fb_sigmoid = (b_sigmoid - x[idx,3])*exp(-b_sigmoid)/(1.0 + exp(-b_sigmoid))^2 - (1.0/(1.0 + exp(-b_sigmoid)) - 1.0/(1.0 + exp(-x[idx,3])))
                
                iter = Int32(1)
                while iter <= Int32(100)
                    if (c_sigmoid - b_sigmoid > b_sigmoid - a_sigmoid)
                        x_sigmoid = b_sigmoid + (2.0 - Base.MathConstants.golden)*(c_sigmoid - b_sigmoid)
                        if abs(c_sigmoid-a_sigmoid) < 1.0e-10*(abs(b_sigmoid) + abs(x_sigmoid)) || iter == Int32(100)
                            xk_cc = (c_sigmoid + a_sigmoid)/2.0
                            break
                        end
                        iter += Int32(1)
                        fx_sigmoid = (x_sigmoid - x[idx,3])*exp(-x_sigmoid)/(1.0 + exp(-x_sigmoid))^2 - (1.0/(1.0 + exp(-x_sigmoid)) - 1.0/(1.0 + exp(-x[idx,3])))
                        if fa_sigmoid*fx_sigmoid < 0.0
                            c_sigmoid = x_sigmoid
                            fc_sigmoid = fx_sigmoid
                        else
                            a_sigmoid = b_sigmoid
                            fa_sigmoid = fb_sigmoid
                            b_sigmoid = x_sigmoid
                            fb_sigmoid = fx_sigmoid
                        end
                    else
                        x_sigmoid = b_sigmoid - (2.0 - Base.MathConstants.golden)*(b_sigmoid - a_sigmoid)
                        if abs(c_sigmoid-a_sigmoid) < 1.0e-10*(abs(b_sigmoid) + abs(x_sigmoid)) || iter == Int32(100)
                            xk_cc = (c_sigmoid + a_sigmoid)/2.0
                            break
                        end
                        iter += Int32(1)
                        fx_sigmoid = (x_sigmoid - x[idx,3])*exp(-x_sigmoid)/(1.0 + exp(-x_sigmoid))^2 - (1.0/(1.0 + exp(-x_sigmoid)) - 1.0/(1.0 + exp(-x[idx,3])))
                        if fa_sigmoid*fb_sigmoid < 0.0
                            c_sigmoid = b_sigmoid
                            fc_sigmoid = fb_sigmoid
                            b_sigmoid = x_sigmoid
                            fb_sigmoid = fx_sigmoid
                        else
                            a_sigmoid = x_sigmoid
                            fa_sigmoid = fx_sigmoid
                        end
                    end
                end
            end

            # Now that we have xk_cv and xk_cc, we can go through the midcv/midcc rules
            if x[idx,2] >= x[idx,1]
                if x[idx,1] >= x[idx,4]
                    if x[idx,1] >= x[idx,3]
                        # Use the results of xk to determine what OUT_cv and dcv should be
                        if x[idx,1] <= xk_cv
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                            dcv = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                        else
                            # cv, dcv = dline_seg(sigmoid, sigmoid_deriv, x[idx,1], p (now xk_cv), xU)
                            if xk_cv == x[idx,4]
                                OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                                dcv = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            else
                                OUT[idx,1] = (1.0/(1.0 + exp(-xk_cv))*(x[idx,4] - x[idx,1]) + 1.0/(1.0 + exp(-x[idx,4]))*(x[idx,1] - xk_cv))/(x[idx,4] - xk_cv)
                                dcv = (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-xk_cv)))/(x[idx,4] - xk_cv)
                            end
                        end
            
                        if x[idx,1] <= xk_cc
                            # cc, dcc = dline_seg(ssigmoid, sigmoid_deriv, x[idx,1], xL, p)
                            if xk_cc == x[idx,3]
                                OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                                dcc = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            else
                                OUT[idx,2] = (1.0/(1.0 + exp(-x[idx,3]))*(xk_cc - x[idx,1]) + 1.0/(1.0 + exp(-xk_cc))*(x[idx,1] - x[idx,3]))/(xk_cc - x[idx,3])
                                dcc = (1.0/(1.0 + exp(-xk_cc)) - 1.0/(1.0 + exp(-x[idx,3])))/(xk_cc - x[idx,3])
                            end
            
                        else
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                            dcc = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                        end
            
                        # Now use the information about dcv and dcc to calculate subgradients
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * dcv
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * dcc
                            col += Int32(1)
                        end
                    elseif x[idx,2] == x[idx,1]
                        # Use the results of xk to determine what OUT_cv and dcv should be
                        if x[idx,1] <= xk_cv
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                            dcv = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                        else
                            # cv, dcv = dline_seg(sigmoid, sigmoid_deriv, x[idx,1], p (now xk_cv), xU)
                            if xk_cv == x[idx,4]
                                OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                                dcv = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            else
                                OUT[idx,1] = (1.0/(1.0 + exp(-xk_cv))*(x[idx,4] - x[idx,1]) + 1.0/(1.0 + exp(-x[idx,4]))*(x[idx,1] - xk_cv))/(x[idx,4] - xk_cv)
                                dcv = (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-xk_cv)))/(x[idx,4] - xk_cv)
                            end
                        end
            
                        if x[idx,1] <= xk_cc
                            # cc, dcc = dline_seg(ssigmoid, sigmoid_deriv, x[idx,1], xL, p)
                            if xk_cc == x[idx,3]
                                OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                                dcc = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            else
                                OUT[idx,2] = (1.0/(1.0 + exp(-x[idx,3]))*(xk_cc - x[idx,1]) + 1.0/(1.0 + exp(-xk_cc))*(x[idx,1] - x[idx,3]))/(xk_cc - x[idx,3])
                                dcc = (1.0/(1.0 + exp(-xk_cc)) - 1.0/(1.0 + exp(-x[idx,3])))/(xk_cc - x[idx,3])
                            end
            
                        else
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                            dcc = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                        end
            
                        # Now use the information about dcv and dcc to calculate subgradients
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * dcv
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * dcc
                            col += Int32(1)
                        end   
                    elseif x[idx,3] >= x[idx,2]
                        # Use the results of xk to determine what OUT_cv and dcv should be
                        if x[idx,2] <= xk_cv
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                            dcv = exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                        else
                            # cv, dcv = dline_seg(sigmoid, sigmoid_deriv, x[idx,2], p (now xk_cv), xU)
                            if xk_cv == x[idx,4]
                                OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                                dcv = exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                            else
                                OUT[idx,1] = (1.0/(1.0 + exp(-xk_cv))*(x[idx,4] - x[idx,2]) + 1.0/(1.0 + exp(-x[idx,4]))*(x[idx,2] - xk_cv))/(x[idx,4] - xk_cv)
                                dcv = (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-xk_cv)))/(x[idx,4] - xk_cv)
                            end
                        end
            
                        if x[idx,1] <= xk_cc
                            # cc, dcc = dline_seg(ssigmoid, sigmoid_deriv, x[idx,1], xL, p)
                            if xk_cc == x[idx,3]
                                OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                                dcc = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            else
                                OUT[idx,2] = (1.0/(1.0 + exp(-x[idx,3]))*(xk_cc - x[idx,1]) + 1.0/(1.0 + exp(-xk_cc))*(x[idx,1] - x[idx,3]))/(xk_cc - x[idx,3])
                                dcc = (1.0/(1.0 + exp(-xk_cc)) - 1.0/(1.0 + exp(-x[idx,3])))/(xk_cc - x[idx,3])
                            end
            
                        else
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                            dcc = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                        end
            
                        # Now use the information about dcv and dcc to calculate subgradients
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * dcv
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * dcc
                            col += Int32(1)
                        end
                    else
                        # Use the results of xk to determine what OUT_cv and dcv should be
                        if x[idx,3] <= xk_cv
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                        else
                            # cv, dcv = dline_seg(sigmoid, sigmoid_deriv, x[idx,3], p (now xk_cv), xU)
                            if xk_cv == x[idx,4]
                                OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                            else
                                OUT[idx,1] = (1.0/(1.0 + exp(-xk_cv))*(x[idx,4] - x[idx,3]) + 1.0/(1.0 + exp(-x[idx,4]))*(x[idx,3] - xk_cv))/(x[idx,4] - xk_cv)
                            end
                        end
            
                        if x[idx,1] <= xk_cc
                            # cc, dcc = dline_seg(ssigmoid, sigmoid_deriv, x[idx,1], xL, p)
                            if xk_cc == x[idx,3]
                                OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                                dcc = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            else
                                OUT[idx,2] = (1.0/(1.0 + exp(-x[idx,3]))*(xk_cc - x[idx,1]) + 1.0/(1.0 + exp(-xk_cc))*(x[idx,1] - x[idx,3]))/(xk_cc - x[idx,3])
                                dcc = (1.0/(1.0 + exp(-xk_cc)) - 1.0/(1.0 + exp(-x[idx,3])))/(xk_cc - x[idx,3])
                            end
            
                        else
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                            dcc = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                        end
            
                        # Now use the information about dcv and dcc to calculate subgradients
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * dcc
                            col += Int32(1)
                        end
                    end
                elseif x[idx,2] == x[idx,1]
                    # Use the results of xk to determine what OUT_cv and dcv should be
                    if x[idx,1] <= xk_cv
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                        dcv = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                    else
                        # cv, dcv = dline_seg(sigmoid, sigmoid_deriv, x[idx,1], p (now xk_cv), xU)
                        if xk_cv == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                            dcv = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                        else
                            OUT[idx,1] = (1.0/(1.0 + exp(-xk_cv))*(x[idx,4] - x[idx,1]) + 1.0/(1.0 + exp(-x[idx,4]))*(x[idx,1] - xk_cv))/(x[idx,4] - xk_cv)
                            dcv = (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-xk_cv)))/(x[idx,4] - xk_cv)
                        end
                    end
        
                    if x[idx,1] <= xk_cc
                        # cc, dcc = dline_seg(ssigmoid, sigmoid_deriv, x[idx,1], xL, p)
                        if xk_cc == x[idx,3]
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                            dcc = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                        else
                            OUT[idx,2] = (1.0/(1.0 + exp(-x[idx,3]))*(xk_cc - x[idx,1]) + 1.0/(1.0 + exp(-xk_cc))*(x[idx,1] - x[idx,3]))/(xk_cc - x[idx,3])
                            dcc = (1.0/(1.0 + exp(-xk_cc)) - 1.0/(1.0 + exp(-x[idx,3])))/(xk_cc - x[idx,3])
                        end
        
                    else
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                        dcc = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                    end
        
                    # Now use the information about dcv and dcc to calculate subgradients
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * dcv
                        OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * dcc
                        col += Int32(1)
                    end  
                elseif x[idx,4] >= x[idx,2]
                    if x[idx,1] >= x[idx,3]
                        # Use the results of xk to determine what OUT_cv and dcv should be
                        if x[idx,1] <= xk_cv
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                            dcv = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                        else
                            # cv, dcv = dline_seg(sigmoid, sigmoid_deriv, x[idx,1], p (now xk_cv), xU)
                            if xk_cv == x[idx,4]
                                OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                                dcv = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            else
                                OUT[idx,1] = (1.0/(1.0 + exp(-xk_cv))*(x[idx,4] - x[idx,1]) + 1.0/(1.0 + exp(-x[idx,4]))*(x[idx,1] - xk_cv))/(x[idx,4] - xk_cv)
                                dcv = (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-xk_cv)))/(x[idx,4] - xk_cv)
                            end
                        end
            
                        if x[idx,2] <= xk_cc
                            # cc, dcc = dline_seg(ssigmoid, sigmoid_deriv, x[idx,2], xL, p)
                            if xk_cc == x[idx,3]
                                OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                                dcc = exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                            else
                                OUT[idx,2] = (1.0/(1.0 + exp(-x[idx,3]))*(xk_cc - x[idx,2]) + 1.0/(1.0 + exp(-xk_cc))*(x[idx,2] - x[idx,3]))/(xk_cc - x[idx,3])
                                dcc = (1.0/(1.0 + exp(-xk_cc)) - 1.0/(1.0 + exp(-x[idx,3])))/(xk_cc - x[idx,3])
                            end
            
                        else
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                            dcc = exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                        end
            
                        # Now use the information about dcv and dcc to calculate subgradients
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * dcv
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * dcc
                            col += Int32(1)
                        end
                    elseif x[idx,3] >= x[idx,2]
                        # Use the results of xk to determine what OUT_cv and dcv should be
                        if x[idx,2] <= xk_cv
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                            dcv = exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                        else
                            # cv, dcv = dline_seg(sigmoid, sigmoid_deriv, x[idx,2], p (now xk_cv), xU)
                            if xk_cv == x[idx,4]
                                OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                                dcv = exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                            else
                                OUT[idx,1] = (1.0/(1.0 + exp(-xk_cv))*(x[idx,4] - x[idx,2]) + 1.0/(1.0 + exp(-x[idx,4]))*(x[idx,2] - xk_cv))/(x[idx,4] - xk_cv)
                                dcv = (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-xk_cv)))/(x[idx,4] - xk_cv)
                            end
                        end
            
                        if x[idx,2] <= xk_cc
                            # cc, dcc = dline_seg(ssigmoid, sigmoid_deriv, x[idx,2], xL, p)
                            if xk_cc == x[idx,3]
                                OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                                dcc = exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                            else
                                OUT[idx,2] = (1.0/(1.0 + exp(-x[idx,3]))*(xk_cc - x[idx,2]) + 1.0/(1.0 + exp(-xk_cc))*(x[idx,2] - x[idx,3]))/(xk_cc - x[idx,3])
                                dcc = (1.0/(1.0 + exp(-xk_cc)) - 1.0/(1.0 + exp(-x[idx,3])))/(xk_cc - x[idx,3])
                            end
            
                        else
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                            dcc = exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                        end
            
                        # Now use the information about dcv and dcc to calculate subgradients
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * dcv
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * dcc
                            col += Int32(1)
                        end
                    else
                        # Use the results of xk to determine what OUT_cv and dcv should be
                        if x[idx,3] <= xk_cv
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                        else
                            # cv, dcv = dline_seg(sigmoid, sigmoid_deriv, x[idx,3], p (now xk_cv), xU)
                            if xk_cv == x[idx,4]
                                OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                            else
                                OUT[idx,1] = (1.0/(1.0 + exp(-xk_cv))*(x[idx,4] - x[idx,3]) + 1.0/(1.0 + exp(-x[idx,4]))*(x[idx,3] - xk_cv))/(x[idx,4] - xk_cv)
                            end
                        end
            
                        if x[idx,2] <= xk_cc
                            # cc, dcc = dline_seg(ssigmoid, sigmoid_deriv, x[idx,2], xL, p)
                            if xk_cc == x[idx,3]
                                OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                                dcc = exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                            else
                                OUT[idx,2] = (1.0/(1.0 + exp(-x[idx,3]))*(xk_cc - x[idx,2]) + 1.0/(1.0 + exp(-xk_cc))*(x[idx,2] - x[idx,3]))/(xk_cc - x[idx,3])
                                dcc = (1.0/(1.0 + exp(-xk_cc)) - 1.0/(1.0 + exp(-x[idx,3])))/(xk_cc - x[idx,3])
                            end
            
                        else
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                            dcc = exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                        end
            
                        # Now use the information about dcv and dcc to calculate subgradients
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * dcc
                            col += Int32(1)
                        end
                    end
                else
                    if x[idx,1] >= x[idx,3]
                        # Use the results of xk to determine what OUT_cv and dcv should be
                        if x[idx,1] <= xk_cv
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                            dcv = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                        else
                            # cv, dcv = dline_seg(sigmoid, sigmoid_deriv, x[idx,1], p (now xk_cv), xU)
                            if xk_cv == x[idx,4]
                                OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                                dcv = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                            else
                                OUT[idx,1] = (1.0/(1.0 + exp(-xk_cv))*(x[idx,4] - x[idx,1]) + 1.0/(1.0 + exp(-x[idx,4]))*(x[idx,1] - xk_cv))/(x[idx,4] - xk_cv)
                                dcv = (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-xk_cv)))/(x[idx,4] - xk_cv)
                            end
                        end
            
                        if x[idx,4] <= xk_cc
                            # cc, dcc = dline_seg(ssigmoid, sigmoid_deriv, x[idx,4], xL, p)
                            if xk_cc == x[idx,3]
                                OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                            else
                                OUT[idx,2] = (1.0/(1.0 + exp(-x[idx,3]))*(xk_cc - x[idx,4]) + 1.0/(1.0 + exp(-xk_cc))*(x[idx,4] - x[idx,3]))/(xk_cc - x[idx,3])
                            end
            
                        else
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                        end
            
                        # Now use the information about dcv and dcc to calculate subgradients
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * dcv
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    elseif x[idx,3] >= x[idx,2]
                        # Use the results of xk to determine what OUT_cv and dcv should be
                        if x[idx,2] <= xk_cv
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                            dcv = exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                        else
                            # cv, dcv = dline_seg(sigmoid, sigmoid_deriv, x[idx,2], p (now xk_cv), xU)
                            if xk_cv == x[idx,4]
                                OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                                dcv = exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                            else
                                OUT[idx,1] = (1.0/(1.0 + exp(-xk_cv))*(x[idx,4] - x[idx,2]) + 1.0/(1.0 + exp(-x[idx,4]))*(x[idx,2] - xk_cv))/(x[idx,4] - xk_cv)
                                dcv = (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-xk_cv)))/(x[idx,4] - xk_cv)
                            end
                        end
            
                        if x[idx,4] <= xk_cc
                            # cc, dcc = dline_seg(ssigmoid, sigmoid_deriv, x[idx,4], xL, p)
                            if xk_cc == x[idx,3]
                                OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                            else
                                OUT[idx,2] = (1.0/(1.0 + exp(-x[idx,3]))*(xk_cc - x[idx,4]) + 1.0/(1.0 + exp(-xk_cc))*(x[idx,4] - x[idx,3]))/(xk_cc - x[idx,3])
                            end
            
                        else
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                        end
            
                        # Now use the information about dcv and dcc to calculate subgradients
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * dcv
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    else
                        # Use the results of xk to determine what OUT_cv and dcv should be
                        if x[idx,3] <= xk_cv
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                        else
                            # cv, dcv = dline_seg(sigmoid, sigmoid_deriv, x[idx,3], p (now xk_cv), xU)
                            if xk_cv == x[idx,4]
                                OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                            else
                                OUT[idx,1] = (1.0/(1.0 + exp(-xk_cv))*(x[idx,4] - x[idx,3]) + 1.0/(1.0 + exp(-x[idx,4]))*(x[idx,3] - xk_cv))/(x[idx,4] - xk_cv)
                            end
                        end
            
                        if x[idx,4] <= xk_cc
                            # cc, dcc = dline_seg(ssigmoid, sigmoid_deriv, x[idx,4], xL, p)
                            if xk_cc == x[idx,3]
                                OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                            else
                                OUT[idx,2] = (1.0/(1.0 + exp(-x[idx,3]))*(xk_cc - x[idx,4]) + 1.0/(1.0 + exp(-xk_cc))*(x[idx,4] - x[idx,3]))/(xk_cc - x[idx,3])
                            end
            
                        else
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                        end
            
                        # Now use the information about dcv and dcc to calculate subgradients
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                end
            elseif x[idx,4] >= x[idx,1]
                if x[idx,3] >= x[idx,1]
                    # Use the results of xk to determine what OUT_cv and dcv should be
                    if x[idx,1] <= xk_cv
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                        dcv = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                    else
                        # cv, dcv = dline_seg(sigmoid, sigmoid_deriv, x[idx,1], p (now xk_cv), xU)
                        if xk_cv == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                            dcv = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                        else
                            OUT[idx,1] = (1.0/(1.0 + exp(-xk_cv))*(x[idx,4] - x[idx,1]) + 1.0/(1.0 + exp(-x[idx,4]))*(x[idx,1] - xk_cv))/(x[idx,4] - xk_cv)
                            dcv = (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-xk_cv)))/(x[idx,4] - xk_cv)
                        end
                    end
        
                    if x[idx,1] <= xk_cc
                        # cc, dcc = dline_seg(ssigmoid, sigmoid_deriv, x[idx,1], xL, p)
                        if xk_cc == x[idx,3]
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                            dcc = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                        else
                            OUT[idx,2] = (1.0/(1.0 + exp(-x[idx,3]))*(xk_cc - x[idx,1]) + 1.0/(1.0 + exp(-xk_cc))*(x[idx,1] - x[idx,3]))/(xk_cc - x[idx,3])
                            dcc = (1.0/(1.0 + exp(-xk_cc)) - 1.0/(1.0 + exp(-x[idx,3])))/(xk_cc - x[idx,3])
                        end
        
                    else
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                        dcc = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                    end
        
                    # Now use the information about dcv and dcc to calculate subgradients
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * dcv
                        OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * dcc
                        col += Int32(1)
                    end
                elseif x[idx,2] >= x[idx,3]
                    # Use the results of xk to determine what OUT_cv and dcv should be
                    if x[idx,2] <= xk_cv
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                        dcv = exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                    else
                        # cv, dcv = dline_seg(sigmoid, sigmoid_deriv, x[idx,2], p (now xk_cv), xU)
                        if xk_cv == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                            dcv = exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                        else
                            OUT[idx,1] = (1.0/(1.0 + exp(-xk_cv))*(x[idx,4] - x[idx,2]) + 1.0/(1.0 + exp(-x[idx,4]))*(x[idx,2] - xk_cv))/(x[idx,4] - xk_cv)
                            dcv = (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-xk_cv)))/(x[idx,4] - xk_cv)
                        end
                    end
        
                    if x[idx,1] <= xk_cc
                        # cc, dcc = dline_seg(ssigmoid, sigmoid_deriv, x[idx,1], xL, p)
                        if xk_cc == x[idx,3]
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                            dcc = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                        else
                            OUT[idx,2] = (1.0/(1.0 + exp(-x[idx,3]))*(xk_cc - x[idx,1]) + 1.0/(1.0 + exp(-xk_cc))*(x[idx,1] - x[idx,3]))/(xk_cc - x[idx,3])
                            dcc = (1.0/(1.0 + exp(-xk_cc)) - 1.0/(1.0 + exp(-x[idx,3])))/(xk_cc - x[idx,3])
                        end
        
                    else
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                        dcc = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                    end
        
                    # Now use the information about dcv and dcc to calculate subgradients
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * dcv
                        OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * dcc
                        col += Int32(1)
                    end
                else
                    # Use the results of xk to determine what OUT_cv and dcv should be
                    if x[idx,3] <= xk_cv
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                    else
                        # cv, dcv = dline_seg(sigmoid, sigmoid_deriv, x[idx,3], p (now xk_cv), xU)
                        if xk_cv == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                        else
                            OUT[idx,1] = (1.0/(1.0 + exp(-xk_cv))*(x[idx,4] - x[idx,3]) + 1.0/(1.0 + exp(-x[idx,4]))*(x[idx,3] - xk_cv))/(x[idx,4] - xk_cv)
                        end
                    end
        
                    if x[idx,1] <= xk_cc
                        # cc, dcc = dline_seg(ssigmoid, sigmoid_deriv, x[idx,1], xL, p)
                        if xk_cc == x[idx,3]
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                            dcc = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                        else
                            OUT[idx,2] = (1.0/(1.0 + exp(-x[idx,3]))*(xk_cc - x[idx,1]) + 1.0/(1.0 + exp(-xk_cc))*(x[idx,1] - x[idx,3]))/(xk_cc - x[idx,3])
                            dcc = (1.0/(1.0 + exp(-xk_cc)) - 1.0/(1.0 + exp(-x[idx,3])))/(xk_cc - x[idx,3])
                        end
        
                    else
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,1]))
                        dcc = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                    end
        
                    # Now use the information about dcv and dcc to calculate subgradients
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = 0.0
                        OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * dcc
                        col += Int32(1)
                    end
                end
            elseif x[idx,2] >= x[idx,4]
                if x[idx,3] >= x[idx,1]
                    # Use the results of xk to determine what OUT_cv and dcv should be
                    if x[idx,1] <= xk_cv
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                        dcv = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                    else
                        # cv, dcv = dline_seg(sigmoid, sigmoid_deriv, x[idx,1], p (now xk_cv), xU)
                        if xk_cv == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                            dcv = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                        else
                            OUT[idx,1] = (1.0/(1.0 + exp(-xk_cv))*(x[idx,4] - x[idx,1]) + 1.0/(1.0 + exp(-x[idx,4]))*(x[idx,1] - xk_cv))/(x[idx,4] - xk_cv)
                            dcv = (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-xk_cv)))/(x[idx,4] - xk_cv)
                        end
                    end
        
                    if x[idx,2] <= xk_cc
                        # cc, dcc = dline_seg(ssigmoid, sigmoid_deriv, x[idx,2], xL, p)
                        if xk_cc == x[idx,3]
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                            dcc = exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                        else
                            OUT[idx,2] = (1.0/(1.0 + exp(-x[idx,3]))*(xk_cc - x[idx,2]) + 1.0/(1.0 + exp(-xk_cc))*(x[idx,2] - x[idx,3]))/(xk_cc - x[idx,3])
                            dcc = (1.0/(1.0 + exp(-xk_cc)) - 1.0/(1.0 + exp(-x[idx,3])))/(xk_cc - x[idx,3])
                        end
        
                    else
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                        dcc = exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                    end
        
                    # Now use the information about dcv and dcc to calculate subgradients
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * dcv
                        OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * dcc
                        col += Int32(1)
                    end
                elseif x[idx,2] >= x[idx,3]
                    # Use the results of xk to determine what OUT_cv and dcv should be
                    if x[idx,2] <= xk_cv
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                        dcv = exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                    else
                        # cv, dcv = dline_seg(sigmoid, sigmoid_deriv, x[idx,2], p (now xk_cv), xU)
                        if xk_cv == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                            dcv = exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                        else
                            OUT[idx,1] = (1.0/(1.0 + exp(-xk_cv))*(x[idx,4] - x[idx,2]) + 1.0/(1.0 + exp(-x[idx,4]))*(x[idx,2] - xk_cv))/(x[idx,4] - xk_cv)
                            dcv = (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-xk_cv)))/(x[idx,4] - xk_cv)
                        end
                    end
        
                    if x[idx,2] <= xk_cc
                        # cc, dcc = dline_seg(ssigmoid, sigmoid_deriv, x[idx,2], xL, p)
                        if xk_cc == x[idx,3]
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                            dcc = exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                        else
                            OUT[idx,2] = (1.0/(1.0 + exp(-x[idx,3]))*(xk_cc - x[idx,2]) + 1.0/(1.0 + exp(-xk_cc))*(x[idx,2] - x[idx,3]))/(xk_cc - x[idx,3])
                            dcc = (1.0/(1.0 + exp(-xk_cc)) - 1.0/(1.0 + exp(-x[idx,3])))/(xk_cc - x[idx,3])
                        end
        
                    else
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                        dcc = exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                    end
        
                    # Now use the information about dcv and dcc to calculate subgradients
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * dcv
                        OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * dcc
                        col += Int32(1)
                    end
                else
                    # Use the results of xk to determine what OUT_cv and dcv should be
                    if x[idx,3] <= xk_cv
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                    else
                        # cv, dcv = dline_seg(sigmoid, sigmoid_deriv, x[idx,3], p (now xk_cv), xU)
                        if xk_cv == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                        else
                            OUT[idx,1] = (1.0/(1.0 + exp(-xk_cv))*(x[idx,4] - x[idx,3]) + 1.0/(1.0 + exp(-x[idx,4]))*(x[idx,3] - xk_cv))/(x[idx,4] - xk_cv)
                        end
                    end
        
                    if x[idx,2] <= xk_cc
                        # cc, dcc = dline_seg(ssigmoid, sigmoid_deriv, x[idx,2], xL, p)
                        if xk_cc == x[idx,3]
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                            dcc = exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                        else
                            OUT[idx,2] = (1.0/(1.0 + exp(-x[idx,3]))*(xk_cc - x[idx,2]) + 1.0/(1.0 + exp(-xk_cc))*(x[idx,2] - x[idx,3]))/(xk_cc - x[idx,3])
                            dcc = (1.0/(1.0 + exp(-xk_cc)) - 1.0/(1.0 + exp(-x[idx,3])))/(xk_cc - x[idx,3])
                        end
        
                    else
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,2]))
                        dcc = exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                    end
        
                    # Now use the information about dcv and dcc to calculate subgradients
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = 0.0
                        OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * dcc
                        col += Int32(1)
                    end
                end
            else
                if x[idx,3] >= x[idx,1]
                    # Use the results of xk to determine what OUT_cv and dcv should be
                    if x[idx,1] <= xk_cv
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                        dcv = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                    else
                        # cv, dcv = dline_seg(sigmoid, sigmoid_deriv, x[idx,1], p (now xk_cv), xU)
                        if xk_cv == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,1]))
                            dcv = exp(-x[idx,1])/(1.0 + exp(-x[idx,1]))^2
                        else
                            OUT[idx,1] = (1.0/(1.0 + exp(-xk_cv))*(x[idx,4] - x[idx,1]) + 1.0/(1.0 + exp(-x[idx,4]))*(x[idx,1] - xk_cv))/(x[idx,4] - xk_cv)
                            dcv = (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-xk_cv)))/(x[idx,4] - xk_cv)
                        end
                    end
        
                    if x[idx,4] <= xk_cc
                        # cc, dcc = dline_seg(ssigmoid, sigmoid_deriv, x[idx,4], xL, p)
                        if xk_cc == x[idx,3]
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                        else
                            OUT[idx,2] = (1.0/(1.0 + exp(-x[idx,3]))*(xk_cc - x[idx,4]) + 1.0/(1.0 + exp(-xk_cc))*(x[idx,4] - x[idx,3]))/(xk_cc - x[idx,3])
                        end
        
                    else
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                    end
        
                    # Now use the information about dcv and dcc to calculate subgradients
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * dcv
                        OUT[idx,end-1*colmax+col] = 0.0
                        col += Int32(1)
                    end
                elseif x[idx,2] >= x[idx,3]
                    # Use the results of xk to determine what OUT_cv and dcv should be
                    if x[idx,2] <= xk_cv
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                        dcv = exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                    else
                        # cv, dcv = dline_seg(sigmoid, sigmoid_deriv, x[idx,2], p (now xk_cv), xU)
                        if xk_cv == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,2]))
                            dcv = exp(-x[idx,2])/(1.0 + exp(-x[idx,2]))^2
                        else
                            OUT[idx,1] = (1.0/(1.0 + exp(-xk_cv))*(x[idx,4] - x[idx,2]) + 1.0/(1.0 + exp(-x[idx,4]))*(x[idx,2] - xk_cv))/(x[idx,4] - xk_cv)
                            dcv = (1.0/(1.0 + exp(-x[idx,4])) - 1.0/(1.0 + exp(-xk_cv)))/(x[idx,4] - xk_cv)
                        end
                    end
        
                    if x[idx,4] <= xk_cc
                        # cc, dcc = dline_seg(ssigmoid, sigmoid_deriv, x[idx,4], xL, p)
                        if xk_cc == x[idx,3]
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                        else
                            OUT[idx,2] = (1.0/(1.0 + exp(-x[idx,3]))*(xk_cc - x[idx,4]) + 1.0/(1.0 + exp(-xk_cc))*(x[idx,4] - x[idx,3]))/(xk_cc - x[idx,3])
                        end
        
                    else
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                    end
        
                    # Now use the information about dcv and dcc to calculate subgradients
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * dcv
                        OUT[idx,end-1*colmax+col] = 0.0
                        col += Int32(1)
                    end
                else
                    # Use the results of xk to determine what OUT_cv and dcv should be
                    if x[idx,3] <= xk_cv
                        OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                    else
                        # cv, dcv = dline_seg(sigmoid, sigmoid_deriv, x[idx,3], p (now xk_cv), xU)
                        if xk_cv == x[idx,4]
                            OUT[idx,1] = 1.0/(1.0 + exp(-x[idx,3]))
                        else
                            OUT[idx,1] = (1.0/(1.0 + exp(-xk_cv))*(x[idx,4] - x[idx,3]) + 1.0/(1.0 + exp(-x[idx,4]))*(x[idx,3] - xk_cv))/(x[idx,4] - xk_cv)
                        end
                    end
        
                    if x[idx,4] <= xk_cc
                        # cc, dcc = dline_seg(ssigmoid, sigmoid_deriv, x[idx,4], xL, p)
                        if xk_cc == x[idx,3]
                            OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                        else
                            OUT[idx,2] = (1.0/(1.0 + exp(-x[idx,3]))*(xk_cc - x[idx,4]) + 1.0/(1.0 + exp(-xk_cc))*(x[idx,4] - x[idx,3]))/(xk_cc - x[idx,3])
                        end
        
                    else
                        OUT[idx,2] = 1.0/(1.0 + exp(-x[idx,4]))
                    end
        
                    # Now use the information about dcv and dcc to calculate subgradients
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = 0.0
                        OUT[idx,end-1*colmax+col] = 0.0
                        col += Int32(1)
                    end
                end
            end
        end

        idx += stride
    end
    return nothing
end

# Positive even integer powers
# max threads: ???
# NOTE: You can reduce the number of temporary variables by combining the various
# if-else blocks into one another, which increases the kernel speed, but at the
# expense of significantly longer compilation times. The following times were
# recorded using my workstation, in the format:
#  Method
#     Compilation/first run time
#     Kernel runtime (8192-length inputs)
#
#  Default (~200 line function)
#     0.751169 seconds (680.11 k CPU allocations: 32.493 MiB)
#     138.600 μs (12 allocations: 256 bytes)
# 
#  Med-length (Keeping eps_min/max in its own if-else tree, combining midcc/cv
#              cc/cv_id and final assignment trees; ~400 line function)
#     7.573736 seconds (3.19 M CPU allocations: 141.522 MiB, 0.46% gc time)
#     136.000 μs (12 allocations: 256 bytes)
#
#  Max-length (Combining eps_min/max, midcc/cv, cc/cv_id, and assignments into
#              one deep if-else tree; ~1470 line function)
#     115.306541 seconds (12.22 M CPU allocations: 529.422 MiB, 0.13% gc time)
#     124.800 μs (12 allocations: 256 bytes)
# 
# (For now, these examples are saved in the even_power_example.jl file)
# For the purposes of using this kernel as a part of a larger kernel for more
# complicated functions, the extra 12 μs being saved individually likely is
# irrelevant in the face of making the kernel much longer. I.e., if the larger
# kernel needs to be split, more individual kernels need to be launched, which
# is a major factor in the overall time. The 12 μs saved here could mean 100 μs
# lost to requiring an additional kernel launch. Thus, I'll be using the short
# version of the kernel.

function SCMC_even_power_kernel(OUT::CuDeviceMatrix, x::CuDeviceMatrix, z::Integer)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    colmax = Int32((size(OUT,2)-4)/2)

    while idx <= Int32(size(OUT,1))
        # Reset the column counter
        col = Int32(1)

        if x[idx,4] <= 0.0
            eps_min = x[idx,4]
            eps_max = x[idx,3]
        elseif x[idx,3] >= 0.0
            eps_min = x[idx,3]
            eps_max = x[idx,4]
        elseif abs(x[idx,3]) >= abs(x[idx,4])
            eps_min = 0.0
            eps_max = x[idx,3]
        else
            eps_min = 0.0
            eps_max = x[idx,4]
        end
        OUT[idx,3] = eps_min^z
        OUT[idx,4] = eps_max^z

        if x[idx,2] >= x[idx,1]
            if x[idx,1] == x[idx,2]
                midcv = x[idx,1] 
                cv_id = Int32(2)
                midcc = x[idx,1] 
                cc_id = Int32(2)
            elseif x[idx,1] >= eps_max
                if x[idx,1] >= eps_min
                    midcv = x[idx,1] 
                    cv_id = Int32(2)
                    midcc = x[idx,1] 
                    cc_id = Int32(2)
                elseif eps_min >= x[idx,2]
                    midcv = x[idx,2] 
                    cv_id = Int32(1)
                    midcc = x[idx,1] 
                    cc_id = Int32(2)
                else
                    midcv = eps_min 
                    cv_id = Int32(3)
                    midcc = x[idx,1] 
                    cc_id = Int32(2)
                end
            elseif eps_max >= x[idx,2]
                if x[idx,1] >= eps_min
                    midcv = x[idx,1] 
                    cv_id = Int32(2)
                    midcc = x[idx,2] 
                    cc_id = Int32(1)
                elseif eps_min >= x[idx,2]
                    midcv = x[idx,2] 
                    cv_id = Int32(1)
                    midcc = x[idx,2] 
                    cc_id = Int32(1)
                else
                    midcv = eps_min 
                    cv_id = Int32(3)
                    midcc = x[idx,2] 
                    cc_id = Int32(1)
                end
            else
                if x[idx,1] >= eps_min
                    midcv = x[idx,1] 
                    cv_id = Int32(2)
                    midcc = eps_max 
                    cc_id = Int32(3)
                elseif eps_min >= x[idx,2]
                    midcv = x[idx,2] 
                    cv_id = Int32(1)
                    midcc = eps_max 
                    cc_id = Int32(3)
                else
                    midcv = eps_min 
                    cv_id = Int32(3)
                    midcc = eps_max 
                    cc_id = Int32(3)
                end
            end
        elseif eps_max >= x[idx,1]
            if eps_min >= x[idx,1]
                midcv = x[idx,1] 
                cv_id = Int32(2)
                midcc = x[idx,1] 
                cc_id = Int32(2)
            elseif x[idx,2] >= eps_min
                midcv = x[idx,2] 
                cv_id = Int32(1)
                midcc = x[idx,1] 
                cc_id = Int32(2)
            else
                midcv = eps_min 
                cv_id = Int32(3)
                midcc = x[idx,1] 
                cc_id = Int32(2)
            end
        elseif x[idx,2] >= eps_max
            if eps_min >= x[idx,1]
                midcv = x[idx,1] 
                cv_id = Int32(2)
                midcc = x[idx,2] 
                cc_id = Int32(1)
            elseif x[idx,2] >= eps_min
                midcv = x[idx,2] 
                cv_id = Int32(1)
                midcc = x[idx,2] 
                cc_id = Int32(1)
            else
                midcv = eps_min 
                cv_id = Int32(3)
                midcc = x[idx,2] 
                cc_id = Int32(1)
            end
        else
            if eps_min >= x[idx,1]
                midcv = x[idx,1] 
                cv_id = Int32(2)
                midcc = eps_max 
                cc_id = Int32(3)
            elseif x[idx,2] >= eps_min
                midcv = x[idx,2] 
                cv_id = Int32(1)
                midcc = eps_max 
                cc_id = Int32(3)
            else
                midcv = eps_min 
                cv_id = Int32(3)
                midcc = eps_max 
                cc_id = Int32(3)
            end
        end

        if x[idx,3] == x[idx,4]
            OUT[idx,1] = midcv*midcv^(z-1)
            OUT[idx,2] = midcc^z
            while col <= colmax
                if cv_id==Int32(1)
                    OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*midcv^(z-1)
                elseif cv_id==Int32(2)
                    OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*midcv^(z-1)
                else
                    OUT[idx,end-2*colmax+col] = 0.0
                end

                if cc_id==Int32(1)
                    OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*midcc^(z-1)
                elseif cc_id==Int32(2)
                    OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*midcc^(z-1)
                else
                    OUT[idx,end-1*colmax+col] = 0.0
                end
                col += Int32(1)
            end
        else
            OUT[idx,1] = midcv*midcv^(z-1)
            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - midcc) + x[idx,4]^z*(midcc - x[idx,3]))/(x[idx,4] - x[idx,3])
            while col <= colmax
                if cv_id==Int32(1)
                    OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*midcv^(z-1)
                elseif cv_id==Int32(2)
                    OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*midcv^(z-1)
                else
                    OUT[idx,end-2*colmax+col] = 0.0
                end
                if cc_id==Int32(1)
                    OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                elseif cc_id==Int32(2)
                    OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                else
                    OUT[idx,end-1*colmax+col] = 0.0
                end
                col += Int32(1)
            end
        end
        
        ############################
        # Perform the cut operation
        if OUT[idx,1] < OUT[idx,3]
            OUT[idx,1] = OUT[idx,3]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-2*colmax+col] = 0.0
                col += Int32(1)
            end
        end
        if OUT[idx,2] > OUT[idx,4]
            OUT[idx,2] = OUT[idx,4]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-1*colmax+col] = 0.0
                col += Int32(1)
            end
        end

        idx += stride
    end
    return nothing
end

# Special case of the even power kernel that should be slightly faster
# (We can also remove exponentiation if the power would be (z-1) == (2-1) == 1.
#  Also, we can apply this simplification to remove some math:
#  (xU^2 - xL^2)/(xU - xL)
#  (xU - xL)*(xU + xL)/(xU - xL)
#  (xU + xL)
function SCMC_squared_kernel(OUT::CuDeviceMatrix, x::CuDeviceMatrix)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    colmax = Int32((size(OUT,2)-4)/2)

    while idx <= Int32(size(OUT,1))
        # Reset the column counter
        col = Int32(1)

        if x[idx,4] <= 0.0
            eps_min = x[idx,4]
            eps_max = x[idx,3]
        elseif x[idx,3] >= 0.0
            eps_min = x[idx,3]
            eps_max = x[idx,4]
        elseif abs(x[idx,3]) >= abs(x[idx,4])
            eps_min = 0.0
            eps_max = x[idx,3]
        else
            eps_min = 0.0
            eps_max = x[idx,4]
        end
        OUT[idx,3] = eps_min^2
        OUT[idx,4] = eps_max^2

        if x[idx,2] >= x[idx,1]
            if x[idx,1] == x[idx,2]
                midcv = x[idx,1] 
                cv_id = Int32(2)
                midcc = x[idx,1] 
                cc_id = Int32(2)
            elseif x[idx,1] >= eps_max
                if x[idx,1] >= eps_min
                    midcv = x[idx,1] 
                    cv_id = Int32(2)
                    midcc = x[idx,1] 
                    cc_id = Int32(2)
                elseif eps_min >= x[idx,2]
                    midcv = x[idx,2] 
                    cv_id = Int32(1)
                    midcc = x[idx,1] 
                    cc_id = Int32(2)
                else
                    midcv = eps_min 
                    cv_id = Int32(3)
                    midcc = x[idx,1] 
                    cc_id = Int32(2)
                end
            elseif eps_max >= x[idx,2]
                if x[idx,1] >= eps_min
                    midcv = x[idx,1] 
                    cv_id = Int32(2)
                    midcc = x[idx,2] 
                    cc_id = Int32(1)
                elseif eps_min >= x[idx,2]
                    midcv = x[idx,2] 
                    cv_id = Int32(1)
                    midcc = x[idx,2] 
                    cc_id = Int32(1)
                else
                    midcv = eps_min 
                    cv_id = Int32(3)
                    midcc = x[idx,2] 
                    cc_id = Int32(1)
                end
            else
                if x[idx,1] >= eps_min
                    midcv = x[idx,1] 
                    cv_id = Int32(2)
                    midcc = eps_max 
                    cc_id = Int32(3)
                elseif eps_min >= x[idx,2]
                    midcv = x[idx,2] 
                    cv_id = Int32(1)
                    midcc = eps_max 
                    cc_id = Int32(3)
                else
                    midcv = eps_min 
                    cv_id = Int32(3)
                    midcc = eps_max 
                    cc_id = Int32(3)
                end
            end
        elseif eps_max >= x[idx,1]
            if eps_min >= x[idx,1]
                midcv = x[idx,1] 
                cv_id = Int32(2)
                midcc = x[idx,1] 
                cc_id = Int32(2)
            elseif x[idx,2] >= eps_min
                midcv = x[idx,2] 
                cv_id = Int32(1)
                midcc = x[idx,1] 
                cc_id = Int32(2)
            else
                midcv = eps_min 
                cv_id = Int32(3)
                midcc = x[idx,1] 
                cc_id = Int32(2)
            end
        elseif x[idx,2] >= eps_max
            if eps_min >= x[idx,1]
                midcv = x[idx,1] 
                cv_id = Int32(2)
                midcc = x[idx,2] 
                cc_id = Int32(1)
            elseif x[idx,2] >= eps_min
                midcv = x[idx,2] 
                cv_id = Int32(1)
                midcc = x[idx,2] 
                cc_id = Int32(1)
            else
                midcv = eps_min 
                cv_id = Int32(3)
                midcc = x[idx,2] 
                cc_id = Int32(1)
            end
        else
            if eps_min >= x[idx,1]
                midcv = x[idx,1] 
                cv_id = Int32(2)
                midcc = eps_max 
                cc_id = Int32(3)
            elseif x[idx,2] >= eps_min
                midcv = x[idx,2] 
                cv_id = Int32(1)
                midcc = eps_max 
                cc_id = Int32(3)
            else
                midcv = eps_min 
                cv_id = Int32(3)
                midcc = eps_max 
                cc_id = Int32(3)
            end
        end

        if x[idx,3] == x[idx,4]
            OUT[idx,1] = midcv*midcv
            OUT[idx,2] = midcc^2
            while col <= colmax
                if cv_id==Int32(1)
                    OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * 2*midcv
                elseif cv_id==Int32(2)
                    OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * 2*midcv
                else
                    OUT[idx,end-2*colmax+col] = 0.0
                end

                if cc_id==Int32(1)
                    OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * 2*midcc
                elseif cc_id==Int32(2)
                    OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * 2*midcc
                else
                    OUT[idx,end-1*colmax+col] = 0.0
                end
                col += Int32(1)
            end
        else
            OUT[idx,1] = midcv*midcv
            OUT[idx,2] = (x[idx,3]^2*(x[idx,4] - midcc) + x[idx,4]^2*(midcc - x[idx,3]))/(x[idx,4] - x[idx,3])
            while col <= colmax
                if cv_id==Int32(1)
                    OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * 2*midcv
                elseif cv_id==Int32(2)
                    OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * 2*midcv
                else
                    OUT[idx,end-2*colmax+col] = 0.0
                end
                if cc_id==Int32(1)
                    OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4] + x[idx,3])
                elseif cc_id==Int32(2)
                    OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4] + x[idx,3])
                else
                    OUT[idx,end-1*colmax+col] = 0.0
                end
                col += Int32(1)
            end
        end
        
        ############################
        # Perform the cut operation
        if OUT[idx,1] < OUT[idx,3]
            OUT[idx,1] = OUT[idx,3]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-2*colmax+col] = 0.0
                col += Int32(1)
            end
        end
        if OUT[idx,2] > OUT[idx,4]
            OUT[idx,2] = OUT[idx,4]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-1*colmax+col] = 0.0
                col += Int32(1)
            end
        end

        idx += stride
    end
    return nothing
end

function SCMC_odd_power_kernel(OUT::CuDeviceMatrix, x::CuDeviceMatrix, z::Integer)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    colmax = Int32((size(OUT,2)-4)/2)

    while idx <= Int32(size(OUT,1))
        # Reset the column counter
        col = Int32(1)

        OUT[idx,3] = x[idx,3]^z
        OUT[idx,4] = x[idx,4]^z

        if x[idx,2] >= x[idx,1]
            if x[idx,1] == x[idx,2]
                midcv = x[idx,1] 
                cv_id = Int32(2)
                midcc = x[idx,1] 
                cc_id = Int32(2)
            elseif x[idx,1] >= x[idx,4]
                if x[idx,1] >= x[idx,3]
                    midcv = x[idx,1] 
                    cv_id = Int32(2)
                    midcc = x[idx,1] 
                    cc_id = Int32(2)
                elseif x[idx,3] >= x[idx,2]
                    midcv = x[idx,2] 
                    cv_id = Int32(1)
                    midcc = x[idx,1] 
                    cc_id = Int32(2)
                else
                    midcv = x[idx,3] 
                    cv_id = Int32(3)
                    midcc = x[idx,1] 
                    cc_id = Int32(2)
                end
            elseif x[idx,4] >= x[idx,2]
                if x[idx,1] >= x[idx,3]
                    midcv = x[idx,1] 
                    cv_id = Int32(2)
                    midcc = x[idx,2] 
                    cc_id = Int32(1)
                elseif x[idx,3] >= x[idx,2]
                    midcv = x[idx,2] 
                    cv_id = Int32(1)
                    midcc = x[idx,2] 
                    cc_id = Int32(1)
                else
                    midcv = x[idx,3] 
                    cv_id = Int32(3)
                    midcc = x[idx,2] 
                    cc_id = Int32(1)
                end
            else
                if x[idx,1] >= x[idx,3]
                    midcv = x[idx,1] 
                    cv_id = Int32(2)
                    midcc = x[idx,4] 
                    cc_id = Int32(3)
                elseif x[idx,3] >= x[idx,2]
                    midcv = x[idx,2] 
                    cv_id = Int32(1)
                    midcc = x[idx,4] 
                    cc_id = Int32(3)
                else
                    midcv = x[idx,3] 
                    cv_id = Int32(3)
                    midcc = x[idx,4] 
                    cc_id = Int32(3)
                end
            end
        elseif x[idx,4] >= x[idx,1]
            if x[idx,3] >= x[idx,1]
                midcv = x[idx,1] 
                cv_id = Int32(2)
                midcc = x[idx,1] 
                cc_id = Int32(2)
            elseif x[idx,2] >= x[idx,3]
                midcv = x[idx,2] 
                cv_id = Int32(1)
                midcc = x[idx,1] 
                cc_id = Int32(2)
            else
                midcv = x[idx,3] 
                cv_id = Int32(3)
                midcc = x[idx,1] 
                cc_id = Int32(2)
            end
        elseif x[idx,2] >= x[idx,4]
            if x[idx,3] >= x[idx,1]
                midcv = x[idx,1] 
                cv_id = Int32(2)
                midcc = x[idx,2] 
                cc_id = Int32(1)
            elseif x[idx,2] >= x[idx,3]
                midcv = x[idx,2] 
                cv_id = Int32(1)
                midcc = x[idx,2] 
                cc_id = Int32(1)
            else
                midcv = x[idx,3] 
                cv_id = Int32(3)
                midcc = x[idx,2] 
                cc_id = Int32(1)
            end
        else
            if x[idx,3] >= x[idx,1]
                midcv = x[idx,1] 
                cv_id = Int32(2)
                midcc = x[idx,4] 
                cc_id = Int32(3)
            elseif x[idx,2] >= x[idx,3]
                midcv = x[idx,2] 
                cv_id = Int32(1)
                midcc = x[idx,4] 
                cc_id = Int32(3)
            else
                midcv = x[idx,3] 
                cv_id = Int32(3)
                midcc = x[idx,4] 
                cc_id = Int32(3)
            end
        end

        # Now we have midcv/cc, cv/cc_id.
        if x[idx,4] <= 0.0
            if x[idx,3]==x[idx,4] # Both are function itself
                OUT[idx,1] = midcv^z
                OUT[idx,2] = midcc^z
                while col <= colmax
                    if cv_id==Int32(1)
                        OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*midcv^(z-1)
                    elseif cv_id==Int32(2)
                        OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*midcv^(z-1)
                    else
                        OUT[idx,end-2*colmax+col] = 0.0
                    end
                    if cc_id==Int32(1)
                        OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*midcc^(z-1)
                    elseif cc_id==Int32(2)
                        OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*midcc^(z-1)
                    else
                        OUT[idx,end-1*colmax+col] = 0.0
                    end
                    col += Int32(1)
                end
            else # cv uses line segment
                OUT[idx,1] = (x[idx,3]^z*(x[idx,4] - midcv) + x[idx,4]^z*(midcv - x[idx,3]))/(x[idx,4] - x[idx,3])
                OUT[idx,2] = midcc^z
                while col <= colmax
                    if cv_id==Int32(1)
                        OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                    elseif cv_id==Int32(2)
                        OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                    else
                        OUT[idx,end-2*colmax+col] = 0.0
                    end
                    if cc_id==Int32(1)
                        OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*midcc^(z-1)
                    elseif cc_id==Int32(2)
                        OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*midcc^(z-1)
                    else
                        OUT[idx,end-1*colmax+col] = 0.0
                    end
                    col += Int32(1)
                end
            end
        elseif 0.0 <= x[idx,3]
            if x[idx,3]==x[idx,4] # Both are function itself
                OUT[idx,1] = midcv^z
                OUT[idx,2] = midcc^z
                while col <= colmax
                    if cv_id==Int32(1)
                        OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*midcv^(z-1)
                    elseif cv_id==Int32(2)
                        OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*midcv^(z-1)
                    else
                        OUT[idx,end-2*colmax+col] = 0.0
                    end
                    if cc_id==Int32(1)
                        OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*midcc^(z-1)
                    elseif cc_id==Int32(2)
                        OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*midcc^(z-1)
                    else
                        OUT[idx,end-1*colmax+col] = 0.0
                    end
                    col += Int32(1)
                end
            else # cc uses line segment
                OUT[idx,1] = midcv^z
                OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - midcc) + x[idx,4]^z*(midcc - x[idx,3]))/(x[idx,4] - x[idx,3])
                while col <= colmax
                    if cv_id==Int32(1)
                        OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*midcv^(z-1)
                    elseif cv_id==Int32(2)
                        OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*midcv^(z-1)
                    else
                        OUT[idx,end-2*colmax+col] = 0.0
                    end
                    if cc_id==Int32(1)
                        OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                    elseif cc_id==Int32(2)
                        OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                    else
                        OUT[idx,end-1*colmax+col] = 0.0
                    end
                    col += Int32(1)
                end
            end
        else
            # The inflection point can be determined analytically.
            # Use these values up to z=15 (could go higher if we need)
            if z==3
                xk_cv = min(-0.5*x[idx,3], x[idx,4])
                xk_cc = max(-0.5*x[idx,4], x[idx,3])
            elseif z==5
                xk_cv = min(-0.6058295861882683*x[idx,3], x[idx,4])
                xk_cc = max(-0.6058295861882683*x[idx,4], x[idx,3])
            elseif z==7
                xk_cv = min(-0.6703320476030968*x[idx,3], x[idx,4])
                xk_cc = max(-0.6703320476030968*x[idx,4], x[idx,3])
            elseif z==9
                xk_cv = min(-0.7145377271673349*x[idx,3], x[idx,4])
                xk_cc = max(-0.7145377271673349*x[idx,4], x[idx,3])
            elseif z==11
                xk_cv = min(-0.7470540748651559*x[idx,3], x[idx,4])
                xk_cc = max(-0.7470540748651559*x[idx,4], x[idx,3])
            elseif z==13
                xk_cv = min(-0.7721416355234655*x[idx,3], x[idx,4])
                xk_cc = max(-0.7721416355234655*x[idx,4], x[idx,3])
            elseif z==15
                xk_cv = min(-0.7921778546056709*x[idx,3], x[idx,4])
                xk_cc = max(-0.7921778546056709*x[idx,4], x[idx,3])
            elseif z==17
                xk_cv = min(-0.8086048978723027*x[idx,3], x[idx,4])
                xk_cc = max(-0.8086048978723027*x[idx,4], x[idx,3])
            elseif z==19
                xk_cv = min(-0.8223534102385287*x[idx,3], x[idx,4])
                xk_cc = max(-0.8223534102385287*x[idx,4], x[idx,3])
            elseif z==21
                xk_cv = min(-0.8340533675507736*x[idx,3], x[idx,4])
                xk_cc = max(-0.8340533675507736*x[idx,4], x[idx,3])
            elseif z==23
                xk_cv = min(-0.8441478047418446*x[idx,3], x[idx,4])
                xk_cc = max(-0.8441478047418446*x[idx,4], x[idx,3])
            elseif z==25
                xk_cv = min(-0.8529581643906964*x[idx,3], x[idx,4])
                xk_cc = max(-0.8529581643906964*x[idx,4], x[idx,3])
            elseif z==27
                xk_cv = min(-0.8607238145679608*x[idx,3], x[idx,4])
                xk_cc = max(-0.8607238145679608*x[idx,4], x[idx,3])
            elseif z==29
                xk_cv = min(-0.8676269762720762*x[idx,3], x[idx,4])
                xk_cc = max(-0.8676269762720762*x[idx,4], x[idx,3])
            elseif z==31
                xk_cv = min(-0.8738090154215446*x[idx,3], x[idx,4])
                xk_cc = max(-0.8738090154215446*x[idx,4], x[idx,3])
            elseif z==33
                xk_cv = min(-0.8793814183583145*x[idx,3], x[idx,4])
                xk_cc = max(-0.8793814183583145*x[idx,4], x[idx,3])
            elseif z==35
                xk_cv = min(-0.8844333818207290*x[idx,3], x[idx,4])
                xk_cc = max(-0.8844333818207290*x[idx,4], x[idx,3])
            elseif z==37
                xk_cv = min(-0.8890371830149935*x[idx,3], x[idx,4])
                xk_cc = max(-0.8890371830149935*x[idx,4], x[idx,3])
            elseif z==39
                xk_cv = min(-0.8932520563312301*x[idx,3], x[idx,4])
                xk_cc = max(-0.8932520563312301*x[idx,4], x[idx,3])
            elseif z==41
                xk_cv = min(-0.8971270424799359*x[idx,3], x[idx,4])
                xk_cc = max(-0.8971270424799359*x[idx,4], x[idx,3])
            elseif z==43
                xk_cv = min(-0.9007031161732270*x[idx,3], x[idx,4])
                xk_cc = max(-0.9007031161732270*x[idx,4], x[idx,3])
            elseif z==45
                xk_cv = min(-0.9040147980608216*x[idx,3], x[idx,4])
                xk_cc = max(-0.9040147980608216*x[idx,4], x[idx,3])
            elseif z==47
                xk_cv = min(-0.9070913919345662*x[idx,3], x[idx,4])
                xk_cc = max(-0.9070913919345662*x[idx,4], x[idx,3])
            elseif z==49
                xk_cv = min(-0.9099579456198456*x[idx,3], x[idx,4])
                xk_cc = max(-0.9099579456198456*x[idx,4], x[idx,3])
            else
                # Apply Newton/golden section methods for the convex part,
                # to get x_cv (the inflection point for the convex relaxation)
                dfk = 0.0
                xk_cv = max(0.0, x[idx,4])
                fk = (xk_cv^z - x[idx,3]^z) - (xk_cv-x[idx,3])*(z)*(xk_cv^(z-1))
                flag = true
                iter = Int32(1)
                while iter <= Int32(100)
                    dfk = (z-1)*z*xk_cv^(z-2)*(x[idx,3]-xk_cv)
                    if abs(fk) < 1e-10
                        flag = false
                        break # use xk_cv
                    end
                    if iszero(dfk)
                        xk_cv = 0.0
                        break # Need to do golden section
                    end
                    if (xk_cv == 0.0) && (fk/dfk > 0.0)
                        flag = false
                        break # use xk_cv
                    elseif (xk_cv == x[idx,4]) && (fk/dfk < 0.0)
                        flag = false
                        break # use xk_cv
                    end
                    xk_cv = max(0.0, min(x[idx,4], xk_cv - fk/dfk))
                    fk = (xk_cv^z - x[idx,3]^z) - (xk_cv-x[idx,3])*(z)*(xk_cv^(z-1))
                    iter += Int32(1)
                end

                # If flag, we need to do golden section instead
                if flag
                    a_golden = x[idx,3]
                    fa_golden = (a_golden^z - x[idx,3]^z) - (a_golden-x[idx,3])*(z)*(a_golden^(z-1))
                    c_golden = x[idx,4]
                    fc_golden = (c_golden^z - x[idx,3]^z) - (c_golden-x[idx,3])*(z)*(c_golden^(z-1))

                    if fa_golden*fc_golden > 0
                        xk_cv = NaN
                    end

                    b_golden = x[idx,4] - (2.0 - Base.MathConstants.golden)*(x[idx,4] - x[idx,3])
                    fb_golden = (b_golden^z - x[idx,3]^z) - (b_golden-x[idx,3])*(z)*(b_golden^(z-1))

                    iter = Int32(1)
                    while iter <= Int32(100)
                        if (c_golden - b_golden > b_golden - a_golden)
                            x_golden = b_golden + (2.0 - Base.MathConstants.golden)*(c_golden - b_golden)
                            if abs(c_golden-a_golden) < 1.0e-10*(abs(b_golden) + abs(x_golden)) || iter == Int32(100)
                                xk_cv = (c_golden + a_golden)/2.0
                                break
                            end
                            iter += Int32(1)
                            fx_golden = (x_golden^z - x[idx,3]^z) - (x_golden-x[idx,3])*(z)*(x_golden^(z-1))
                            if fa_golden*fx_golden < 0.0
                                c_golden = x_golden
                                fc_golden = fx_golden
                            else
                                a_golden = b_golden
                                fa_golden = fb_golden
                                b_golden = x_golden
                                fb_golden = fx_golden
                            end
                        else
                            x_golden = b_golden - (2.0 - Base.MathConstants.golden)*(b_golden - a_golden)
                            if abs(c_golden-a_golden) < 1.0e-10*(abs(b_golden) + abs(x_golden)) || iter == Int32(100)
                                xk_cv = (c_golden + a_golden)/2.0
                                break
                            end
                            iter += Int32(1)
                            fx_golden = (x_golden^z - x[idx,3]^z) - (x_golden-x[idx,3])*(z)*(x_golden^(z-1))
                            if fa_golden*fb_golden < 0.0
                                c_golden = b_golden
                                fc_golden = fb_golden
                                b_golden = x_golden
                                fb_golden = fx_golden
                            else
                                a_golden = x_golden
                                fa_golden = fx_golden
                            end
                        end
                    end
                end

                # Apply Newton/golden section methods for the concave part,
                # to get x_cc (the inflection point for the concave relaxation)
                dfk = 0.0
                xk_cc = x[idx,3]
                fk = (x[idx,4]^z-xk_cc^z) - (x[idx,4]-xk_cc)*(z)*(xk_cc^(z-1))
                flag = true
                iter = Int32(1)
                while iter <= Int32(100)
                    dfk = (z-1)*z*xk_cc^(z-2)*(xk_cc-x[idx,4]);
                    if abs(fk) < 1e-10
                        flag = false
                        break # use xk_cc
                    end
                    if iszero(dfk)
                        xk_cc = 0.0
                        break # Need to do golden section
                    end
                    if (xk_cc == x[idx,3]) && (fk/dfk > 0.0)
                        flag = false
                        break # use xk_cc
                    elseif (xk_cc == 0.0) && (fk/dfk < 0.0)
                        flag = false
                        break # use xk_cc
                    end
                    xk_cc = max(x[idx,3], min(0.0, xk_cc - fk/dfk))
                    fk = (x[idx,4]^z-xk_cc^z) - (x[idx,4]-xk_cc)*(z)*(xk_cc^(z-1))
                    iter += Int32(1)
                end

                # If flag, we need to do golden section instead
                if flag
                    a_golden = x[idx,3]
                    fa_golden = (x[idx,4]^z-x[idx,3]^z) - (x[idx,4]-x[idx,3])*(z)*(x[idx,3]^(z-1))
                    c_golden = x[idx,4]
                    fc_golden = 0.0

                    if fa_golden*fc_golden > 0
                        xk_cc = NaN
                    end

                    b_golden = x[idx,4] - (2.0 - Base.MathConstants.golden)*(x[idx,4] - x[idx,3])
                    fb_golden = (x[idx,4]^z-b_golden^z) - (x[idx,4]-b_golden)*(z)*(b_golden^(z-1))

                    iter = Int32(1)
                    while iter <= Int32(100)
                        if (c_golden - b_golden > b_golden - a_golden)
                            x_golden = b_golden + (2.0 - Base.MathConstants.golden)*(c_golden - b_golden)
                            if abs(c_golden-a_golden) < 1.0e-10*(abs(b_golden) + abs(x_golden)) || iter == Int32(100)
                                xk_cc = (c_golden + a_golden)/2.0
                                break
                            end
                            iter += Int32(1)
                            fx_golden = (x[idx,4]^z-x_golden^z) - (x[idx,4]-x_golden)*(z)*(x_golden^(z-1))
                            if fa_golden*fx_golden < 0.0
                                c_golden = x_golden
                                fc_golden = fx_golden
                            else
                                a_golden = b_golden
                                fa_golden = fb_golden
                                b_golden = x_golden
                                fb_golden = fx_golden
                            end
                        else
                            x_golden = b_golden - (2.0 - Base.MathConstants.golden)*(b_golden - a_golden)
                            if abs(c_golden-a_golden) < 1.0e-10*(abs(b_golden) + abs(x_golden)) || iter == Int32(100)
                                xk_cc = (c_golden + a_golden)/2.0
                                break
                            end
                            iter += Int32(1)
                            fx_golden = (x[idx,4]^z-x_golden^z) - (x[idx,4]-x_golden)*(z)*(x_golden^(z-1))
                            if fa_golden*fb_golden < 0.0
                                c_golden = b_golden
                                fc_golden = fb_golden
                                b_golden = x_golden
                                fb_golden = fx_golden
                            else
                                a_golden = x_golden
                                fa_golden = fx_golden
                            end
                        end
                    end
                end
            end

            # Now we have the inflection points, so we either
            # look at the line segment or the function itself
            if (midcv <= xk_cv) && (x[idx,3] != xk_cv) # cv uses line segment
                OUT[idx,1] = (x[idx,3]^z*(xk_cv - midcv) + xk_cv^z*(midcv - x[idx,3]))/(xk_cv - x[idx,3])
                while col <= colmax
                    if cv_id==Int32(1)
                        OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * (xk_cv^z - x[idx,3]^z)/(xk_cv - x[idx,3])
                    elseif cv_id==Int32(2)
                        OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * (xk_cv^z - x[idx,3]^z)/(xk_cv - x[idx,3])
                    else
                        OUT[idx,end-2*colmax+col] = 0.0
                    end
                    col += Int32(1)
                end
            else # cv uses the function itself
                OUT[idx,1] = midcv^z
                while col <= colmax
                    if cv_id==Int32(1)
                        OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*midcv^(z-1)
                    elseif cv_id==Int32(2)
                        OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*midcv^(z-1)
                    else
                        OUT[idx,end-2*colmax+col] = 0.0
                    end
                    col += Int32(1)
                end
            end

            # Reset the column counter
            col = Int32(1)
            
            # Now do the cc side
            if (midcc > xk_cc) && (x[idx,4] != xk_cc) # cc uses line segment
                OUT[idx,2] = (xk_cc^z*(x[idx,4] - midcc) + x[idx,4]^z*(midcc - xk_cc))/(x[idx,4] - xk_cc)
                while col <= colmax
                    if cc_id==Int32(1)
                        OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - xk_cc^z)/(x[idx,4] - xk_cc)
                    elseif cc_id==Int32(2)
                        OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - xk_cc^z)/(x[idx,4] - xk_cc)
                    else
                        OUT[idx,end-1*colmax+col] = 0.0
                    end
                    col += Int32(1)
                end
            else # cc uses the function itself
                OUT[idx,2] = midcc^z
                while col <= colmax
                    if cc_id==Int32(1)
                        OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*midcc^(z-1)
                    elseif cc_id==Int32(2)
                        OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*midcc^(z-1)
                    else
                        OUT[idx,end-1*colmax+col] = 0.0
                    end
                    col += Int32(1)
                end
            end
        end

        ############################
        # Perform the cut operation
        if OUT[idx,1] < OUT[idx,3]
            OUT[idx,1] = OUT[idx,3]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-2*colmax+col] = 0.0
                col += Int32(1)
            end
        end
        if OUT[idx,2] > OUT[idx,4]
            OUT[idx,2] = OUT[idx,4]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-1*colmax+col] = 0.0
                col += Int32(1)
            end
        end

        idx += stride
    end
    return nothing
end


# Power function for small floating-point values. This function is ONLY VALID FOR
# 0 < c < 1 AND xL > 0, NOT FOR ANYTHING OUTSIDE THIS RANGE. 
function SCMC_small_float_power_kernel(OUT::CuDeviceMatrix, x::CuDeviceMatrix, c::T) where T<:Real
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    colmax = Int32((size(OUT,2)-4)/2)

    while idx <= Int32(size(OUT,1))
        # Reset the column counter
        col = Int32(1)

        if x[idx,3] < 0.0
            error("This kernel is not meant to be used for negative lower bounded variables")
        end

        OUT[idx,3] = x[idx,3]^c
        OUT[idx,4] = x[idx,4]^c
        if x[idx,2] >= x[idx,1]
            if x[idx,1] == x[idx,2]
                midcv = x[idx,1] 
                cv_id = Int32(2)
                midcc = x[idx,1] 
                cc_id = Int32(2)
            elseif x[idx,1] >= x[idx,4]
                if x[idx,1] >= x[idx,3]
                    midcv = x[idx,1] 
                    cv_id = Int32(2)
                    midcc = x[idx,1] 
                    cc_id = Int32(2)
                elseif x[idx,3] >= x[idx,2]
                    midcv = x[idx,2] 
                    cv_id = Int32(1)
                    midcc = x[idx,1] 
                    cc_id = Int32(2)
                else
                    midcv = x[idx,3] 
                    cv_id = Int32(3)
                    midcc = x[idx,1] 
                    cc_id = Int32(2)
                end
            elseif x[idx,4] >= x[idx,2]
                if x[idx,1] >= x[idx,3]
                    midcv = x[idx,1] 
                    cv_id = Int32(2)
                    midcc = x[idx,2] 
                    cc_id = Int32(1)
                elseif x[idx,3] >= x[idx,2]
                    midcv = x[idx,2] 
                    cv_id = Int32(1)
                    midcc = x[idx,2] 
                    cc_id = Int32(1)
                else
                    midcv = x[idx,3] 
                    cv_id = Int32(3)
                    midcc = x[idx,2] 
                    cc_id = Int32(1)
                end
            else
                if x[idx,1] >= x[idx,3]
                    midcv = x[idx,1] 
                    cv_id = Int32(2)
                    midcc = x[idx,4] 
                    cc_id = Int32(3)
                elseif x[idx,3] >= x[idx,2]
                    midcv = x[idx,2] 
                    cv_id = Int32(1)
                    midcc = x[idx,4] 
                    cc_id = Int32(3)
                else
                    midcv = x[idx,3] 
                    cv_id = Int32(3)
                    midcc = x[idx,4] 
                    cc_id = Int32(3)
                end
            end
        elseif x[idx,4] >= x[idx,1]
            if x[idx,3] >= x[idx,1]
                midcv = x[idx,1] 
                cv_id = Int32(2)
                midcc = x[idx,1] 
                cc_id = Int32(2)
            elseif x[idx,2] >= x[idx,3]
                midcv = x[idx,2] 
                cv_id = Int32(1)
                midcc = x[idx,1] 
                cc_id = Int32(2)
            else
                midcv = x[idx,3] 
                cv_id = Int32(3)
                midcc = x[idx,1] 
                cc_id = Int32(2)
            end
        elseif x[idx,2] >= x[idx,4]
            if x[idx,3] >= x[idx,1]
                midcv = x[idx,1] 
                cv_id = Int32(2)
                midcc = x[idx,2] 
                cc_id = Int32(1)
            elseif x[idx,2] >= x[idx,3]
                midcv = x[idx,2] 
                cv_id = Int32(1)
                midcc = x[idx,2] 
                cc_id = Int32(1)
            else
                midcv = x[idx,3] 
                cv_id = Int32(3)
                midcc = x[idx,2] 
                cc_id = Int32(1)
            end
        else
            if x[idx,3] >= x[idx,1]
                midcv = x[idx,1] 
                cv_id = Int32(2)
                midcc = x[idx,4] 
                cc_id = Int32(3)
            elseif x[idx,2] >= x[idx,3]
                midcv = x[idx,2] 
                cv_id = Int32(1)
                midcc = x[idx,4] 
                cc_id = Int32(3)
            else
                midcv = x[idx,3] 
                cv_id = Int32(3)
                midcc = x[idx,4] 
                cc_id = Int32(3)
            end
        end

        if x[idx,3] == x[idx,4]
            OUT[idx,1] = midcv^c
            OUT[idx,2] = midcc^c
            while col <= colmax
                if cv_id==Int32(1)
                    OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * c*midcv^(c-1)
                elseif cv_id==Int32(2)
                    OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * c*midcv^(c-1)
                else
                    OUT[idx,end-2*colmax+col] = 0.0
                end

                if cc_id==Int32(1)
                    OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * c*midcc^(c-1)
                elseif cc_id==Int32(2)
                    OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * c*midcc^(c-1)
                else
                    OUT[idx,end-1*colmax+col] = 0.0
                end
                col += Int32(1)
            end
        else
            OUT[idx,1] = (x[idx,3]^c*(x[idx,4] - midcv) + x[idx,4]^c*(midcv - x[idx,3]))/(x[idx,4] - x[idx,3])
            OUT[idx,2] = midcc^c
            while col <= colmax
                if cv_id==Int32(1)
                    OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^c - x[idx,3]^c)/(x[idx,4] - x[idx,3])
                elseif cv_id==Int32(2)
                    OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^c - x[idx,3]^c)/(x[idx,4] - x[idx,3])
                else
                    OUT[idx,end-2*colmax+col] = 0.0
                end

                if cc_id==Int32(1)
                    OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * c*midcc^(c-1)
                elseif cc_id==Int32(2)
                    OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * c*midcc^(c-1)
                else
                    OUT[idx,end-1*colmax+col] = 0.0
                end
                col += Int32(1)
            end
        end
        
        ############################
        # Perform the cut operation
        if OUT[idx,1] < OUT[idx,3]
            OUT[idx,1] = OUT[idx,3]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-2*colmax+col] = 0.0
                col += Int32(1)
            end
        end
        if OUT[idx,2] > OUT[idx,4]
            OUT[idx,2] = OUT[idx,4]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-1*colmax+col] = 0.0
                col += Int32(1)
            end
        end

        idx += stride
    end
    return nothing
end

# Special case of floating-point power rule for 0.5. Some math can be simplified.
function SCMC_sqrt_kernel(OUT::CuDeviceMatrix, x::CuDeviceMatrix)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    colmax = Int32((size(OUT,2)-4)/2)

    while idx <= Int32(size(OUT,1))
        # Reset the column counter
        col = Int32(1)

        OUT[idx,3] = sqrt(x[idx,3])
        OUT[idx,4] = sqrt(x[idx,4])

        if x[idx,2] >= x[idx,1]
            if x[idx,1] == x[idx,2]
                midcv = x[idx,1] 
                cv_id = Int32(2)
                midcc = x[idx,1] 
                cc_id = Int32(2)
            elseif x[idx,1] >= x[idx,4]
                if x[idx,1] >= x[idx,3]
                    midcv = x[idx,1] 
                    cv_id = Int32(2)
                    midcc = x[idx,1] 
                    cc_id = Int32(2)
                elseif x[idx,3] >= x[idx,2]
                    midcv = x[idx,2] 
                    cv_id = Int32(1)
                    midcc = x[idx,1] 
                    cc_id = Int32(2)
                else
                    midcv = x[idx,3] 
                    cv_id = Int32(3)
                    midcc = x[idx,1] 
                    cc_id = Int32(2)
                end
            elseif x[idx,4] >= x[idx,2]
                if x[idx,1] >= x[idx,3]
                    midcv = x[idx,1] 
                    cv_id = Int32(2)
                    midcc = x[idx,2] 
                    cc_id = Int32(1)
                elseif x[idx,3] >= x[idx,2]
                    midcv = x[idx,2] 
                    cv_id = Int32(1)
                    midcc = x[idx,2] 
                    cc_id = Int32(1)
                else
                    midcv = x[idx,3] 
                    cv_id = Int32(3)
                    midcc = x[idx,2] 
                    cc_id = Int32(1)
                end
            else
                if x[idx,1] >= x[idx,3]
                    midcv = x[idx,1] 
                    cv_id = Int32(2)
                    midcc = x[idx,4] 
                    cc_id = Int32(3)
                elseif x[idx,3] >= x[idx,2]
                    midcv = x[idx,2] 
                    cv_id = Int32(1)
                    midcc = x[idx,4] 
                    cc_id = Int32(3)
                else
                    midcv = x[idx,3] 
                    cv_id = Int32(3)
                    midcc = x[idx,4] 
                    cc_id = Int32(3)
                end
            end
        elseif x[idx,4] >= x[idx,1]
            if x[idx,3] >= x[idx,1]
                midcv = x[idx,1] 
                cv_id = Int32(2)
                midcc = x[idx,1] 
                cc_id = Int32(2)
            elseif x[idx,2] >= x[idx,3]
                midcv = x[idx,2] 
                cv_id = Int32(1)
                midcc = x[idx,1] 
                cc_id = Int32(2)
            else
                midcv = x[idx,3] 
                cv_id = Int32(3)
                midcc = x[idx,1] 
                cc_id = Int32(2)
            end
        elseif x[idx,2] >= x[idx,4]
            if x[idx,3] >= x[idx,1]
                midcv = x[idx,1] 
                cv_id = Int32(2)
                midcc = x[idx,2] 
                cc_id = Int32(1)
            elseif x[idx,2] >= x[idx,3]
                midcv = x[idx,2] 
                cv_id = Int32(1)
                midcc = x[idx,2] 
                cc_id = Int32(1)
            else
                midcv = x[idx,3] 
                cv_id = Int32(3)
                midcc = x[idx,2] 
                cc_id = Int32(1)
            end
        else
            if x[idx,3] >= x[idx,1]
                midcv = x[idx,1] 
                cv_id = Int32(2)
                midcc = x[idx,4] 
                cc_id = Int32(3)
            elseif x[idx,2] >= x[idx,3]
                midcv = x[idx,2] 
                cv_id = Int32(1)
                midcc = x[idx,4] 
                cc_id = Int32(3)
            else
                midcv = x[idx,3] 
                cv_id = Int32(3)
                midcc = x[idx,4] 
                cc_id = Int32(3)
            end
        end

        if x[idx,3] == x[idx,4]
            OUT[idx,1] = sqrt(x[idx,3])
            OUT[idx,2] = sqrt(midcc)
            while col <= colmax
                OUT[idx,end-2*colmax+col] = 0.0 # Why? Different from other powers. E.g., x^0.5 != sqrt(x)?

                if cc_id==Int32(1)
                    OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * 0.5/sqrt(x[idx,1])
                elseif cc_id==Int32(2)
                    OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * 0.5/sqrt(x[idx,1])
                else
                    OUT[idx,end-1*colmax+col] = 0.0
                end
                col += Int32(1)
            end
        else
            OUT[idx,1] = ((sqrt(x[idx,4]) - sqrt(x[idx,3]))/(x[idx,4] - x[idx,3]))*(midcv - x[idx,3]) + sqrt(x[idx,3])
            OUT[idx,2] = sqrt(midcc)
            while col <= colmax
                if cv_id==Int32(1)
                    OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * ((sqrt(x[idx,4]) - sqrt(x[idx,3]))/(x[idx,4] - x[idx,3]))
                elseif cv_id==Int32(2)
                    OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * ((sqrt(x[idx,4]) - sqrt(x[idx,3]))/(x[idx,4] - x[idx,3]))
                else
                    OUT[idx,end-2*colmax+col] = 0.0
                end

                if cc_id==Int32(1)
                    OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * 0.5/sqrt(x[idx,1])
                elseif cc_id==Int32(2)
                    OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * 0.5/sqrt(x[idx,1])
                else
                    OUT[idx,end-1*colmax+col] = 0.0
                end
                col += Int32(1)
            end

        end
        
        ############################
        # Perform the cut operation
        if OUT[idx,1] < OUT[idx,3]
            OUT[idx,1] = OUT[idx,3]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-2*colmax+col] = 0.0
                col += Int32(1)
            end
        end
        if OUT[idx,2] > OUT[idx,4]
            OUT[idx,2] = OUT[idx,4]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-1*colmax+col] = 0.0
                col += Int32(1)
            end
        end

        idx += stride
    end
    return nothing
end

# Power function for large floating-point values. This function is ONLY MEANT FOR
# USE OUTSIDE OF 0 < c < 1. For that, use SCMC_small_float_power_kernel.
function SCMC_large_float_power_kernel(OUT::CuDeviceMatrix, x::CuDeviceMatrix, c::T) where T<:Real
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    colmax = Int32((size(OUT,2)-4)/2)

    if isinteger(c)
        error("c is an integer, but it's calling a floating-point kernel")
    end
    
    while idx <= Int32(size(OUT,1))
        # Reset the column counter
        col = Int32(1)

        if x[idx,3] <= 0.0
            OUT[idx,1] = NaN
            OUT[idx,2] = NaN
            OUT[idx,3] = NaN
            OUT[idx,4] = NaN
            while col <= colmax
                OUT[idx,end-2*colmax+col] = NaN
                OUT[idx,end-1*colmax+col] = NaN
                col += Int32(1)
            end
        elseif x[idx,3]==x[idx,4] # Both are function itself
            # Should be that x[idx,1]==x[idx,2]==x[idx,3]==x[idx,4],
            # so the choice of which to use in each case is arbitrary
            OUT[idx,1] = x[idx,1]^c
            OUT[idx,2] = x[idx,2]^c
            while col <= colmax
                if cv_id==Int32(1)
                    OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * c*x[idx,1]^(c-1)
                elseif cv_id==Int32(2)
                    OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * c*x[idx,1]^(c-1)
                else
                    OUT[idx,end-2*colmax+col] = 0.0
                end
                if cc_id==Int32(1)
                    OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * c*x[idx,2]^(c-1)
                elseif cc_id==Int32(2)
                    OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * c*x[idx,2]^(c-1)
                else
                    OUT[idx,end-1*colmax+col] = 0.0
                end
                col += Int32(1)
            end
        else
            if x[idx,2] >= x[idx,1]
                if x[idx,1] == x[idx,2]
                    midcv = x[idx,1] 
                    cv_id = Int32(2)
                    midcc = x[idx,1] 
                    cc_id = Int32(2)
                elseif x[idx,1] >= x[idx,4]
                    if x[idx,1] >= x[idx,3]
                        midcv = x[idx,1] 
                        cv_id = Int32(2)
                        midcc = x[idx,1] 
                        cc_id = Int32(2)
                    elseif x[idx,3] >= x[idx,2]
                        midcv = x[idx,2] 
                        cv_id = Int32(1)
                        midcc = x[idx,1] 
                        cc_id = Int32(2)
                    else
                        midcv = x[idx,3] 
                        cv_id = Int32(3)
                        midcc = x[idx,1] 
                        cc_id = Int32(2)
                    end
                elseif x[idx,4] >= x[idx,2]
                    if x[idx,1] >= x[idx,3]
                        midcv = x[idx,1] 
                        cv_id = Int32(2)
                        midcc = x[idx,2] 
                        cc_id = Int32(1)
                    elseif x[idx,3] >= x[idx,2]
                        midcv = x[idx,2] 
                        cv_id = Int32(1)
                        midcc = x[idx,2] 
                        cc_id = Int32(1)
                    else
                        midcv = x[idx,3] 
                        cv_id = Int32(3)
                        midcc = x[idx,2] 
                        cc_id = Int32(1)
                    end
                else
                    if x[idx,1] >= x[idx,3]
                        midcv = x[idx,1] 
                        cv_id = Int32(2)
                        midcc = x[idx,4] 
                        cc_id = Int32(3)
                    elseif x[idx,3] >= x[idx,2]
                        midcv = x[idx,2] 
                        cv_id = Int32(1)
                        midcc = x[idx,4] 
                        cc_id = Int32(3)
                    else
                        midcv = x[idx,3] 
                        cv_id = Int32(3)
                        midcc = x[idx,4] 
                        cc_id = Int32(3)
                    end
                end
            elseif x[idx,4] >= x[idx,1]
                if x[idx,3] >= x[idx,1]
                    midcv = x[idx,1] 
                    cv_id = Int32(2)
                    midcc = x[idx,1] 
                    cc_id = Int32(2)
                elseif x[idx,2] >= x[idx,3]
                    midcv = x[idx,2] 
                    cv_id = Int32(1)
                    midcc = x[idx,1] 
                    cc_id = Int32(2)
                else
                    midcv = x[idx,3] 
                    cv_id = Int32(3)
                    midcc = x[idx,1] 
                    cc_id = Int32(2)
                end
            elseif x[idx,2] >= x[idx,4]
                if x[idx,3] >= x[idx,1]
                    midcv = x[idx,1] 
                    cv_id = Int32(2)
                    midcc = x[idx,2] 
                    cc_id = Int32(1)
                elseif x[idx,2] >= x[idx,3]
                    midcv = x[idx,2] 
                    cv_id = Int32(1)
                    midcc = x[idx,2] 
                    cc_id = Int32(1)
                else
                    midcv = x[idx,3] 
                    cv_id = Int32(3)
                    midcc = x[idx,2] 
                    cc_id = Int32(1)
                end
            else
                if x[idx,3] >= x[idx,1]
                    midcv = x[idx,1] 
                    cv_id = Int32(2)
                    midcc = x[idx,4] 
                    cc_id = Int32(3)
                elseif x[idx,2] >= x[idx,3]
                    midcv = x[idx,2] 
                    cv_id = Int32(1)
                    midcc = x[idx,4] 
                    cc_id = Int32(3)
                else
                    midcv = x[idx,3] 
                    cv_id = Int32(3)
                    midcc = x[idx,4] 
                    cc_id = Int32(3)
                end
            end

            # Now we have midcv/cc, cv/cc_id.
            # cv uses the function itself, cc uses a line segment between the bounds
            OUT[idx,1] = midcv^c
            OUT[idx,2] = (x[idx,3]^c*(x[idx,4] - midcc) + x[idx,4]^c*(midcc - x[idx,3]))/(x[idx,4] - x[idx,3])
            while col <= colmax
                if cv_id==Int32(1)
                    OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * c*midcv^(c-1)
                elseif cv_id==Int32(2)
                    OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * c*midcv^(c-1)
                else
                    OUT[idx,end-2*colmax+col] = 0.0
                end
                if cc_id==Int32(1)
                    OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^c - x[idx,3]^c)/(x[idx,4] - x[idx,3])
                elseif cc_id==Int32(2)
                    OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^c - x[idx,3]^c)/(x[idx,4] - x[idx,3])
                else
                    OUT[idx,end-1*colmax+col] = 0.0
                end
                col += Int32(1)
            end
        end
        
        ############################
        # Perform the cut operation
        if OUT[idx,1] < OUT[idx,3]
            OUT[idx,1] = OUT[idx,3]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-2*colmax+col] = 0.0
                col += Int32(1)
            end
        end
        if OUT[idx,2] > OUT[idx,4]
            OUT[idx,2] = OUT[idx,4]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-1*colmax+col] = 0.0
                col += Int32(1)
            end
        end

        idx += stride
    end
    return nothing
end

#=
Binary Rules
=#
# Multiplication of two variables
# max threads: 384
function SCMC_mult_kernel(OUT::CuDeviceMatrix, x::CuDeviceMatrix, y::CuDeviceMatrix)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    colmax = Int32((size(OUT,2)-4)/2)

    while idx <= Int32(size(OUT,1))
        # Reset the column counter
        col = Int32(1)

        # Begin multiplication
        if x[idx,3] >= 0.0
            if y[idx,3] >= 0.0
                # x and y strictly positive
                OUT[idx,3] = x[idx,3]*y[idx,3]
                OUT[idx,4] = x[idx,4]*y[idx,4]
                t1_cv_left = y[idx,4]*x[idx,1] + x[idx,4]*y[idx,1] - x[idx,4]*y[idx,4]
                t1_cv_right = y[idx,3]*x[idx,1] + x[idx,3]*y[idx,1] - x[idx,3]*y[idx,3]
                t1_cc_left = y[idx,3]*x[idx,2] + x[idx,4]*y[idx,2] - x[idx,4]*y[idx,3]
                t1_cc_right = y[idx,4]*x[idx,2] + x[idx,3]*y[idx,2] - x[idx,3]*y[idx,4]
                if t1_cv_left > t1_cv_right
                    OUT[idx,1] = t1_cv_left
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = y[idx,4]*x[idx,end-2*colmax+col] + x[idx,4]*y[idx,end-2*colmax+col]
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = t1_cv_right
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = y[idx,3]*x[idx,end-2*colmax+col] + x[idx,3]*y[idx,end-2*colmax+col]
                        col += Int32(1)
                    end
                end
                col = Int32(1)
                if t1_cc_left < t1_cc_right
                    OUT[idx,2] = t1_cc_left
                    while col <= colmax
                        OUT[idx,end-1*colmax+col] = y[idx,3]*x[idx,end-1*colmax+col] + x[idx,4]*y[idx,end-1*colmax+col]
                        col += Int32(1)
                    end
                else
                    OUT[idx,2] = t1_cc_right
                    while col <= colmax
                        OUT[idx,end-1*colmax+col] = y[idx,4]*x[idx,end-1*colmax+col] + x[idx,3]*y[idx,end-1*colmax+col]
                        col += Int32(1)
                    end
                end
            elseif y[idx,4] <= 0.0
                # x strictly positive, y strictly negative
                OUT[idx,3] = x[idx,4]*y[idx,3]
                OUT[idx,4] = x[idx,3]*y[idx,4]
                t1_cv_left = (-y[idx,4])*x[idx,2] + x[idx,4]*(-y[idx,1]) - x[idx,4]*(-y[idx,4])
                t1_cv_right = (-y[idx,3])*x[idx,2] + x[idx,3]*(-y[idx,1]) - x[idx,3]*(-y[idx,3])
                t1_cc_left = (-y[idx,3])*x[idx,1] + x[idx,4]*(-y[idx,2]) - x[idx,4]*(-y[idx,3])
                t1_cc_right = (-y[idx,4])*x[idx,1] + x[idx,3]*(-y[idx,2]) - x[idx,3]*(-y[idx,4])
                if t1_cv_left < t1_cv_right
                    OUT[idx,1] = -t1_cv_left
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = -((-y[idx,4])*x[idx,end-1*colmax+col] + x[idx,4]*(-y[idx,end-2*colmax+col]))
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = -t1_cv_right
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = -((-y[idx,3])*x[idx,end-1*colmax+col] + x[idx,3]*(-y[idx,end-2*colmax+col]))
                        col += Int32(1)
                    end
                end
                col = Int32(1)
                if t1_cc_left > t1_cc_right
                    OUT[idx,2] = -t1_cc_left
                    while col <= colmax
                        OUT[idx,end-1*colmax+col] = -((-y[idx,3])*x[idx,end-2*colmax+col] + x[idx,4]*(-y[idx,end-1*colmax+col]))
                        col += Int32(1)
                    end
                else
                    OUT[idx,2] = -t1_cc_right
                    while col <= colmax
                        OUT[idx,end-1*colmax+col] = -((-y[idx,4])*x[idx,end-2*colmax+col] + x[idx,3]*(-y[idx,end-1*colmax+col]))
                        col += Int32(1)
                    end
                end
            else
                # x strictly positive, y mixed
                OUT[idx,3] = x[idx,4]*y[idx,3]
                OUT[idx,4] = x[idx,4]*y[idx,4]
                t1_cv_left = y[idx,4]*x[idx,1] + x[idx,4]*y[idx,1] - x[idx,4]*y[idx,4]
                t1_cv_right = y[idx,3]*x[idx,2] + x[idx,3]*y[idx,1] - x[idx,3]*y[idx,3]
                t1_cc_left = y[idx,3]*x[idx,1] + x[idx,4]*y[idx,2] - x[idx,4]*y[idx,3]
                t1_cc_right = y[idx,4]*x[idx,2] + x[idx,3]*y[idx,2] - x[idx,3]*y[idx,4]
                if t1_cv_left > t1_cv_right
                    OUT[idx,1] = t1_cv_left
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = y[idx,4]*x[idx,end-2*colmax+col] + x[idx,4]*y[idx,end-2*colmax+col]
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = t1_cv_right
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = y[idx,3]*x[idx,end-1*colmax+col] + x[idx,3]*y[idx,end-2*colmax+col]
                        col += Int32(1)
                    end
                end
                col = Int32(1)
                if t1_cc_left < t1_cc_right
                    OUT[idx,2] = t1_cc_left
                    while col <= colmax
                        OUT[idx,end-1*colmax+col] = y[idx,3]*x[idx,end-2*colmax+col] + x[idx,4]*y[idx,end-1*colmax+col]
                        col += Int32(1)
                    end
                else
                    OUT[idx,2] = t1_cc_right
                    while col <= colmax
                        OUT[idx,end-1*colmax+col] = y[idx,4]*x[idx,end-1*colmax+col] + x[idx,3]*y[idx,end-1*colmax+col]
                        col += Int32(1)
                    end
                end
            end
        elseif x[idx,4] <= 0.0
            if y[idx,3] >= 0.0
                # x strictly negative, y strictly positive
                OUT[idx,3] = x[idx,3]*y[idx,4]
                OUT[idx,4] = x[idx,4]*y[idx,3]
                t1_cv_left = y[idx,3]*(-x[idx,1]) + (-x[idx,3])*y[idx,2] - (-x[idx,3])*y[idx,3]
                t1_cv_right = y[idx,4]*(-x[idx,1]) + (-x[idx,4])*y[idx,2] - (-x[idx,4])*y[idx,4]
                t1_cc_left = y[idx,4]*(-x[idx,2]) + (-x[idx,3])*y[idx,1] - (-x[idx,3])*y[idx,4]
                t1_cc_right = y[idx,3]*(-x[idx,2]) + (-x[idx,4])*y[idx,1] - (-x[idx,4])*y[idx,3]
                if t1_cv_left < t1_cv_right
                    OUT[idx,1] = -t1_cv_left
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = -(y[idx,3]*(-x[idx,end-2*colmax+col]) + (-x[idx,3])*y[idx,end-1*colmax+col])
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = -t1_cv_right
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = -(y[idx,4]*(-x[idx,end-2*colmax+col]) + (-x[idx,4])*y[idx,end-1*colmax+col])
                        col += Int32(1)
                    end
                end
                col = Int32(1)
                if t1_cc_left > t1_cc_right
                    OUT[idx,2] = -t1_cc_left
                    while col <= colmax
                        OUT[idx,end-1*colmax+col] = -(y[idx,4]*(-x[idx,end-1*colmax+col]) + (-x[idx,3])*y[idx,end-2*colmax+col])
                        col += Int32(1)
                    end
                else
                    OUT[idx,2] = -t1_cc_right
                    while col <= colmax
                        OUT[idx,end-1*colmax+col] = -(y[idx,3]*(-x[idx,end-1*colmax+col]) + (-x[idx,4])*y[idx,end-2*colmax+col])
                        col += Int32(1)
                    end
                end
            elseif y[idx,4] <= 0.0
                # x and y strictly negative
                OUT[idx,3] = x[idx,4]*y[idx,4]
                OUT[idx,4] = x[idx,3]*y[idx,3]
                t1_cv_left = y[idx,3]*x[idx,2] + x[idx,3]*y[idx,2] - x[idx,3]*y[idx,3]
                t1_cv_right = y[idx,4]*x[idx,2] + x[idx,4]*y[idx,2] - x[idx,4]*y[idx,4]
                t1_cc_left = y[idx,4]*x[idx,1] + x[idx,3]*y[idx,1] - x[idx,3]*y[idx,4]
                t1_cc_right = y[idx,3]*x[idx,1] + x[idx,4]*y[idx,1] - x[idx,4]*y[idx,3]
                if t1_cv_left > t1_cv_right
                    OUT[idx,1] = t1_cv_left
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = (-y[idx,3])*(-x[idx,end-1*colmax+col]) + (-x[idx,3])*(-y[idx,end-1*colmax+col])
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = t1_cv_right
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = (-y[idx,4])*(-x[idx,end-1*colmax+col]) + (-x[idx,4])*(-y[idx,end-1*colmax+col])
                        col += Int32(1)
                    end
                end
                col = Int32(1)
                if t1_cc_left < t1_cc_right
                    OUT[idx,2] = t1_cc_left
                    while col <= colmax
                        OUT[idx,end-1*colmax+col] = (-y[idx,4])*(-x[idx,end-2*colmax+col]) + (-x[idx,3])*(-y[idx,end-2*colmax+col])
                        col += Int32(1)
                    end
                else
                    OUT[idx,2] = t1_cc_right
                    while col <= colmax
                        OUT[idx,end-1*colmax+col] = (-y[idx,3])*(-x[idx,end-2*colmax+col]) + (-x[idx,4])*(-y[idx,end-2*colmax+col])
                        col += Int32(1)
                    end
                end
            else
                # x strictly negative, y mixed
                OUT[idx,3] = x[idx,3]*y[idx,4]
                OUT[idx,4] = x[idx,3]*y[idx,3]
                t1_cv_left = y[idx,3]*(-x[idx,2]) + (-x[idx,3])*y[idx,2] - (-x[idx,3])*y[idx,3]
                t1_cv_right = y[idx,4]*(-x[idx,1]) + (-x[idx,4])*y[idx,2] - (-x[idx,4])*y[idx,4]
                t1_cc_left = y[idx,4]*(-x[idx,2]) + (-x[idx,3])*y[idx,1] - (-x[idx,3])*y[idx,4]
                t1_cc_right = y[idx,3]*(-x[idx,1]) + (-x[idx,4])*y[idx,1] - (-x[idx,4])*y[idx,3]
                if t1_cv_left < t1_cv_right
                    OUT[idx,1] = -t1_cv_left
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = -(y[idx,3]*(-x[idx,end-1*colmax+col]) + (-x[idx,3])*y[idx,end-1*colmax+col])
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = -t1_cv_right
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = -(y[idx,4]*(-x[idx,end-2*colmax+col]) + (-x[idx,4])*y[idx,end-1*colmax+col])
                        col += Int32(1)
                    end
                end
                col = Int32(1)
                if t1_cc_left > t1_cc_right
                    OUT[idx,2] = -t1_cc_left
                    while col <= colmax
                        OUT[idx,end-1*colmax+col] = -((-x[idx,3])*(y[idx,end-2*colmax+col]) + (y[idx,4])*(-x[idx,end-1*colmax+col]))
                        col += Int32(1)
                    end
                else
                    OUT[idx,2] = -t1_cc_right
                    while col <= colmax
                        OUT[idx,end-1*colmax+col] = -((-x[idx,4])*(y[idx,end-2*colmax+col]) + (y[idx,3])*(-x[idx,end-2*colmax+col]))
                        col += Int32(1)
                    end
                end
            end
        else
            if y[idx,3] >= 0.0
                # x mixed, y strictly positive
                OUT[idx,3] = x[idx,3]*y[idx,4]
                OUT[idx,4] = x[idx,4]*y[idx,4]
                t1_cv_left = x[idx,4]*y[idx,1] + y[idx,4]*x[idx,1] - y[idx,4]*x[idx,4]
                t1_cv_right = x[idx,3]*y[idx,2] + y[idx,3]*x[idx,1] - y[idx,3]*x[idx,3]
                t1_cc_left = x[idx,3]*y[idx,1] + y[idx,4]*x[idx,2] - y[idx,4]*x[idx,3]
                t1_cc_right = x[idx,4]*y[idx,2] + y[idx,3]*x[idx,2] - y[idx,3]*x[idx,4]
                if t1_cv_left > t1_cv_right
                    OUT[idx,1] = t1_cv_left
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = (x[idx,4])*(y[idx,end-2*colmax+col]) + (y[idx,4])*(x[idx,end-2*colmax+col])
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = t1_cv_right
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = (x[idx,3])*(y[idx,end-1*colmax+col]) + (y[idx,3])*(x[idx,end-2*colmax+col])
                        col += Int32(1)
                    end
                end
                col = Int32(1)
                if t1_cc_left < t1_cc_right
                    OUT[idx,2] = t1_cc_left
                    while col <= colmax
                        OUT[idx,end-1*colmax+col] = (x[idx,3])*(y[idx,end-2*colmax+col]) + (y[idx,4])*(x[idx,end-1*colmax+col])
                        col += Int32(1)
                    end
                else
                    OUT[idx,2] = t1_cc_right
                    while col <= colmax
                        OUT[idx,end-1*colmax+col] = (x[idx,4])*(y[idx,end-1*colmax+col]) + (y[idx,3])*(x[idx,end-1*colmax+col])
                        col += Int32(1)
                    end
                end
            elseif y[idx,4] <= 0.0
                # x mixed, y strictly negative
                OUT[idx,3] = x[idx,4]*y[idx,3]
                OUT[idx,4] = x[idx,3]*y[idx,3]
                t1_cv_left = x[idx,3]*(-y[idx,2]) + (-y[idx,3])*x[idx,2] - (-y[idx,3])*x[idx,3]
                t1_cv_right = x[idx,4]*(-y[idx,1]) + (-y[idx,4])*x[idx,2] - (-y[idx,4])*x[idx,4]
                t1_cc_left = x[idx,4]*(-y[idx,2]) + (-y[idx,3])*x[idx,1] - (-y[idx,3])*x[idx,4]
                t1_cc_right = x[idx,3]*(-y[idx,1]) + (-y[idx,4])*x[idx,1] - (-y[idx,4])*x[idx,3]
                if t1_cv_left < t1_cv_right
                    OUT[idx,1] = -t1_cv_left
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = -((x[idx,3])*(-y[idx,end-1*colmax+col]) + (-y[idx,3])*(x[idx,end-1*colmax+col]))
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = -t1_cv_right
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = -((x[idx,4])*(-y[idx,end-2*colmax+col]) + (-y[idx,4])*(x[idx,end-1*colmax+col]))
                        col += Int32(1)
                    end
                end
                col = Int32(1)
                if t1_cc_left > t1_cc_right
                    OUT[idx,2] = -t1_cc_left
                    while col <= colmax
                        OUT[idx,end-1*colmax+col] = -((x[idx,4])*(-y[idx,end-1*colmax+col]) + (-y[idx,3])*(x[idx,end-2*colmax+col]))
                        col += Int32(1)
                    end
                else
                    OUT[idx,2] = -t1_cc_right
                    while col <= colmax
                        OUT[idx,end-1*colmax+col] = -((x[idx,3])*(-y[idx,end-2*colmax+col]) + (-y[idx,4])*(x[idx,end-2*colmax+col]))
                        col += Int32(1)
                    end
                end
            else
                # x and y both mixed
                OUT[idx,3] = min(x[idx,3]*y[idx,4], x[idx,4]*y[idx,3])
                OUT[idx,4] = max(x[idx,3]*y[idx,3], x[idx,4]*y[idx,4])
                t1_cv_left = y[idx,4]*x[idx,1] + x[idx,4]*y[idx,1] - x[idx,4]*y[idx,4]
                t1_cv_right = y[idx,3]*x[idx,2] + x[idx,3]*y[idx,2] - x[idx,3]*y[idx,3]
                t1_cc_left = y[idx,3]*x[idx,1] + x[idx,4]*y[idx,2] - x[idx,4]*y[idx,3]
                t1_cc_right = y[idx,4]*x[idx,2] + x[idx,3]*y[idx,1] - x[idx,3]*y[idx,4]
                if t1_cv_left > t1_cv_right
                    OUT[idx,1] = t1_cv_left
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = y[idx,4]*x[idx,end-2*colmax+col] + x[idx,4]*y[idx,end-2*colmax+col]
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = t1_cv_right
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = y[idx,3]*x[idx,end-1*colmax+col] + x[idx,3]*y[idx,end-1*colmax+col]
                        col += Int32(1)
                    end
                end
                col = Int32(1)
                if t1_cc_left < t1_cc_right
                    OUT[idx,2] = t1_cc_left
                    while col <= colmax
                        OUT[idx,end-1*colmax+col] = y[idx,3]*x[idx,end-2*colmax+col] + x[idx,4]*y[idx,end-1*colmax+col]
                        col += Int32(1)
                    end
                else
                    OUT[idx,2] = t1_cc_right
                    while col <= colmax
                        OUT[idx,end-1*colmax+col] = y[idx,4]*x[idx,end-1*colmax+col] + x[idx,3]*y[idx,end-2*colmax+col]
                        col += Int32(1)
                    end
                end
            end
        end

        # Perform the cut operation
        if OUT[idx,1] < OUT[idx,3]
            OUT[idx,1] = OUT[idx,3]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-2*colmax+col] = 0.0
                col += Int32(1)
            end
        end
        if OUT[idx,2] > OUT[idx,4]
            OUT[idx,2] = OUT[idx,4]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-1*colmax+col] = 0.0
                col += Int32(1)
            end
        end

        idx += stride
    end
    return nothing
end

# Addition of two variables
# max threads: 1024
function SCMC_add_kernel(OUT::CuDeviceMatrix, x::CuDeviceMatrix, y::CuDeviceMatrix)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    colmax = Int32((size(OUT,2)-4)/2)

    while idx <= Int32(size(OUT,1))
        # Reset the column counter
        col = Int32(1)

        # Begin addition
        OUT[idx,1] = x[idx,1] + y[idx,1]
        OUT[idx,2] = x[idx,2] + y[idx,2]
        OUT[idx,3] = x[idx,3] + y[idx,3]
        OUT[idx,4] = x[idx,4] + y[idx,4]
        while col <= colmax
            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] + y[idx,end-2*colmax+col]
            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] + y[idx,end-1*colmax+col]
            col += Int32(1)
        end
        
        # Perform the cut operation
        if OUT[idx,1] < OUT[idx,3]
            OUT[idx,1] = OUT[idx,3]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-2*colmax+col] = 0.0
                col += Int32(1)
            end
        end
        if OUT[idx,2] > OUT[idx,4]
            OUT[idx,2] = OUT[idx,4]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-1*colmax+col] = 0.0
                col += Int32(1)
            end
        end

        idx += stride
    end
    return nothing
end



##################
# Some templates that are useful for writing new kernels.

function TEMPLATE_KERNEL(OUT::CuDeviceMatrix, x::CuDeviceMatrix, z::Integer)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    colmax = Int32((size(OUT,2)-4)/2)

    while idx <= Int32(size(OUT,1))
        # Reset the column counter
        col = Int32(1)

        if x[idx,4] <= 0.0
            eps_min = x[idx,4]
            eps_max = x[idx,3]
        elseif x[idx,3] >= 0.0
            eps_min = x[idx,3]
            eps_max = x[idx,4]
        elseif abs(x[idx,3]) >= abs(x[idx,4])
            eps_min = 0.0
            eps_max = x[idx,3]
        else
            eps_min = 0.0
            eps_max = x[idx,4]
        end

        if xcc >= xcv
            if xcv == xcc
                midcc = xcv 
                cc_id = Int32(2)
                midcv = xcv 
                cv_id = Int32(2)
            elseif xcv >= eps_max
                if xcv >= eps_min
                    midcc = xcv 
                    cc_id = Int32(2)
                    midcv = xcv 
                    cv_id = Int32(2)
                elseif eps_min >= xcc
                    midcc = xcv 
                    cc_id = Int32(2)
                    midcv = xcc 
                    cv_id = Int32(1)
                else
                    midcc = xcv 
                    cc_id = Int32(2)
                    midcv = eps_min 
                    cv_id = Int32(3)
                end
            elseif eps_max >= xcc
                if xcv >= eps_min
                    midcc = xcc 
                    cc_id = Int32(1)
                    midcv = xcv 
                    cv_id = Int32(2)
                elseif eps_min >= xcc
                    midcc = xcc 
                    cc_id = Int32(1)
                    midcv = xcc 
                    cv_id = Int32(1)
                else
                    midcc = xcc 
                    cc_id = Int32(1)
                    midcv = eps_min 
                    cv_id = Int32(3)
                end
            else
                if xcv >= eps_min
                    midcc = eps_max 
                    cc_id = Int32(3)
                    midcv = xcv 
                    cv_id = Int32(2)
                elseif eps_min >= xcc
                    midcc = eps_max 
                    cc_id = Int32(3)
                    midcv = xcc 
                    cv_id = Int32(1)
                else
                    midcc = eps_max 
                    cc_id = Int32(3)
                    midcv = eps_min 
                    cv_id = Int32(3)
                end
            end
        elseif eps_max >= xcv
            if eps_min >= xcv
                midcc = xcv 
                cc_id = Int32(2)
                midcv = xcv 
                cv_id = Int32(2)
            elseif xcc >= eps_min
                midcc = xcv 
                cc_id = Int32(2)
                midcv = xcc 
                cv_id = Int32(1)
            else
                midcc = xcv 
                cc_id = Int32(2)
                midcv = eps_min 
                cv_id = Int32(3)
            end
        elseif xcc >= eps_max
            if eps_min >= xcv
                midcc = xcc 
                cc_id = Int32(1)
                midcv = xcv 
                cv_id = Int32(2)
            elseif xcc >= eps_min
                midcc = xcc 
                cc_id = Int32(1)
                midcv = xcc 
                cv_id = Int32(1)
            else
                midcc = xcc 
                cc_id = Int32(1)
                midcv = eps_min 
                cv_id = Int32(3)
            end
        else
            if eps_min >= xcv
                midcc = eps_max 
                cc_id = Int32(3)
                midcv = xcv 
                cv_id = Int32(2)
            elseif xcc >= eps_min
                midcc = eps_max 
                cc_id = Int32(3)
                midcv = xcc 
                cv_id = Int32(1)
            else
                midcc = eps_max 
                cc_id = Int32(3)
                midcv = eps_min 
                cv_id = Int32(3)
            end
        end

        cv = midcv*midcv^(z-1)
        dcv = z*midcv*^(z-1)
        if xL == xU
            cc = midcc^z
            dcc = z*midcc^(z-1)
        else
            cc = (xL^z*(xU - midcc) + xU^z*(midcc - xL))/(xU - xL)
            dcc = (xU^z - zL^z)/(xU - xL)
        end

        while col <= colmax
            OUT[idx,end-2*colmax+col] = cv_id * dcv
            OUT[idx,end-1*colmax+col] = cc_id * dcc
            col += Int32(1)
        end


        # Perform the cut operation
        if OUT[idx,1] < OUT[idx,3]
            OUT[idx,1] = OUT[idx,3]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-2*colmax+col] = 0.0
                col += Int32(1)
            end
        end
        if OUT[idx,2] > OUT[idx,4]
            OUT[idx,2] = OUT[idx,4]
            col = Int32(1)
            while col <= colmax
                OUT[idx,end-1*colmax+col] = 0.0
                col += Int32(1)
            end
        end

        idx += stride
    end
    return nothing
end

function newton_or_golden_section_TEMPLATE(x0, xL, xU, f, df, envp1, envp2)
    dfk = 0.0
    xk_CV_OR_CC = max(xL, min(x0, xU))
    fk = f(xk, envp1, envp2)
    flag = true
    iter = Int32(1)
    while iter <= Int32(100)
        dfk = df(xk_CV_OR_CC, envp1, envp2)
        if abs(fk) < 1e-10
            flag = false
            break # use xk_CV_OR_CC
        end
        if iszero(dfk)
            xk_CV_OR_CC = 0.0
            break # Need to do golden section
        end
        if (xk_CV_OR_CC == xL) && (fk/dfk > 0.0)
            flag = false
            break # use xk_CV_OR_CC
        elseif (xk_CV_OR_CC == xU) && (fk/dfk < 0.0)
            flag = false
            break # use xk_CV_OR_CC
        end
        xk_CV_OR_CC = max(xL, min(xU, xk_CV_OR_CC - dk/dfk))
        fk = f(xk_CV_OR_CC, envp1, envp2)
        iter += Int32(1)
    end

    # If flag, we need to do golden section instead
    if flag
        a_golden = xL
        fa_golden = f(a_golden, envp1, envp2)
        c_golden = xU
        fc_golden = f(c_golden, envp1, envp2)

        if fa_golden*fc_golden > 0
            xk_CV_OR_CC = NaN
        end

        b_golden = xU - (2.0 - Base.MathConstants.golden)*(xU - xL)
        fb_golden = f(b_golden, envp1, envp2)

        iter = Int32(1)
        while iter <= Int32(100)
            if (c_golden - b_golden > b_golden - a_golden)
                x_golden = b_golden + (2.0 - Base.MathConstants.golden)*(c_golden - b_golden)
                if abs(c_golden-a_golden) < 1.0e-10*(abs(b_golden) + abs(x_golden)) || iter == Int32(100)
                    xk_CV_OR_CC = (c_golden + a_golden)/2.0
                    break
                end
                iter += Int32(1)
                fx_golden = f(x_golden, envp1, envp2)
                if fa_golden*fx_golden < 0.0
                    c_golden = x_golden
                    fc_golden = fx_golden
                else
                    a_golden = b_golden
                    fa_golden = fb_golden
                    b_golden = x_golden
                    fb_golden = fx_golden
                end
            else
                x_golden = b_golden - (2.0 - Base.MathConstants.golden)*(b_golden - a_golden)
                if abs(c_golden-a_golden) < 1.0e-10*(abs(b_golden) + abs(x_golden)) || iter == Int32(100)
                    xk_CV_OR_CC = (c_golden + a_golden)/2.0
                    break
                end
                iter += Int32(1)
                fx_golden = f(x_golden, envp1, envp2)
                if fa_golden*fb_golden < 0.0
                    c_golden = b_golden
                    fc_golden = fb_golden
                    b_golden = x_golden
                    fb_golden = fx_golden
                else
                    a_golden = x_golden
                    fa_golden = fx_golden
                end
            end
        end
    end
end