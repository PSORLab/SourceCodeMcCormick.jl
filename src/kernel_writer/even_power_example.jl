# Positive even integer powers

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

function SCMC_even_power_kernel_short(OUT::CuDeviceMatrix, x::CuDeviceMatrix, z::Integer)
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
                midcc = x[idx,1] 
                cc_id = 2
                midcv = x[idx,1] 
                cv_id = 2
            elseif x[idx,1] >= eps_max
                if x[idx,1] >= eps_min
                    midcc = x[idx,1] 
                    cc_id = 2
                    midcv = x[idx,1] 
                    cv_id = 2
                elseif eps_min >= x[idx,2]
                    midcc = x[idx,1] 
                    cc_id = 2
                    midcv = x[idx,2] 
                    cv_id = 1
                else
                    midcc = x[idx,1] 
                    cc_id = 2
                    midcv = eps_min 
                    cv_id = 3
                end
            elseif eps_max >= x[idx,2]
                if x[idx,1] >= eps_min
                    midcc = x[idx,2] 
                    cc_id = 1
                    midcv = x[idx,1] 
                    cv_id = 2
                elseif eps_min >= x[idx,2]
                    midcc = x[idx,2] 
                    cc_id = 1
                    midcv = x[idx,2] 
                    cv_id = 1
                else
                    midcc = x[idx,2] 
                    cc_id = 1
                    midcv = eps_min 
                    cv_id = 3
                end
            else
                if x[idx,1] >= eps_min
                    midcc = eps_max 
                    cc_id = 3
                    midcv = x[idx,1] 
                    cv_id = 2
                elseif eps_min >= x[idx,2]
                    midcc = eps_max 
                    cc_id = 3
                    midcv = x[idx,2] 
                    cv_id = 1
                else
                    midcc = eps_max 
                    cc_id = 3
                    midcv = eps_min 
                    cv_id = 3
                end
            end
        elseif eps_max >= x[idx,1]
            if eps_min >= x[idx,1]
                midcc = x[idx,1] 
                cc_id = 2
                midcv = x[idx,1] 
                cv_id = 2
            elseif x[idx,2] >= eps_min
                midcc = x[idx,1] 
                cc_id = 2
                midcv = x[idx,2] 
                cv_id = 1
            else
                midcc = x[idx,1] 
                cc_id = 2
                midcv = eps_min 
                cv_id = 3
            end
        elseif x[idx,2] >= eps_max
            if eps_min >= x[idx,1]
                midcc = x[idx,2] 
                cc_id = 1
                midcv = x[idx,1] 
                cv_id = 2
            elseif x[idx,2] >= eps_min
                midcc = x[idx,2] 
                cc_id = 1
                midcv = x[idx,2] 
                cv_id = 1
            else
                midcc = x[idx,2] 
                cc_id = 1
                midcv = eps_min 
                cv_id = 3
            end
        else
            if eps_min >= x[idx,1]
                midcc = eps_max 
                cc_id = 3
                midcv = x[idx,1] 
                cv_id = 2
            elseif x[idx,2] >= eps_min
                midcc = eps_max 
                cc_id = 3
                midcv = x[idx,2] 
                cv_id = 1
            else
                midcc = eps_max 
                cc_id = 3
                midcv = eps_min 
                cv_id = 3
            end
        end

        if x[idx,3] == x[idx,4]
            OUT[idx,1] = midcv*midcv^(z-1)
            OUT[idx,2] = midcc^z
            while col <= colmax
                if cv_id==1
                    OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*midcv^(z-1)
                elseif cv_id==2
                    OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*midcv^(z-1)
                else
                    OUT[idx,end-2*colmax+col] = 0.0
                end

                if cc_id==1
                    OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*midcc^(z-1)
                elseif cc_id==2
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
                if cv_id==1
                    OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*midcv^(z-1)
                elseif cv_id==2
                    OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*midcv^(z-1)
                else
                    OUT[idx,end-2*colmax+col] = 0.0
                end
                if cc_id==1
                    OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                elseif cc_id==2
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

function SCMC_even_power_kernel_medium(OUT::CuDeviceMatrix, x::CuDeviceMatrix, z::Integer)
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
                if x[idx,3] == x[idx,4]
                    OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                    OUT[idx,2] = x[idx,1]^z
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                        OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                    OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                        OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                        col += Int32(1)
                    end
                end
            elseif x[idx,1] >= eps_max
                if x[idx,1] >= eps_min
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = x[idx,1]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                elseif eps_min >= x[idx,2]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = x[idx,1]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                else
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = eps_min*eps_min^(z-1)
                        OUT[idx,2] = x[idx,1]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = eps_min*eps_min^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                end
            elseif eps_max >= x[idx,2]
                if x[idx,1] >= eps_min
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = x[idx,2]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                elseif eps_min >= x[idx,2]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = x[idx,2]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                else
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = eps_min*eps_min^(z-1)
                        OUT[idx,2] = x[idx,2]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = eps_min*eps_min^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                end
            else
                if x[idx,1] >= eps_min
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = eps_max^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - eps_max) + x[idx,4]^z*(eps_max - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                elseif eps_min >= x[idx,2]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = eps_max^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - eps_max) + x[idx,4]^z*(eps_max - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                else
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = eps_min*eps_min^(z-1)
                        OUT[idx,2] = eps_max^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = eps_min*eps_min^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - eps_max) + x[idx,4]^z*(eps_max - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                end
            end
        elseif eps_max >= x[idx,1]
            if eps_min >= x[idx,1]
                if x[idx,3] == x[idx,4]
                    OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                    OUT[idx,2] = x[idx,1]^z
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                        OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                    OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                        OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                        col += Int32(1)
                    end
                end
            elseif x[idx,2] >= eps_min
                if x[idx,3] == x[idx,4]
                    OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                    OUT[idx,2] = x[idx,1]^z
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                        OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                    OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                        OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                        col += Int32(1)
                    end
                end
            else
                if x[idx,3] == x[idx,4]
                    OUT[idx,1] = eps_min*eps_min^(z-1)
                    OUT[idx,2] = x[idx,1]^z
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = 0.0
                        OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = eps_min*eps_min^(z-1)
                    OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = 0.0
                        OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                        col += Int32(1)
                    end
                end
            end
        elseif x[idx,2] >= eps_max
            if eps_min >= x[idx,1]
                if x[idx,3] == x[idx,4]
                    OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                    OUT[idx,2] = x[idx,2]^z
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                        OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                    OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                        OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                        col += Int32(1)
                    end
                end
            elseif x[idx,2] >= eps_min
                if x[idx,3] == x[idx,4]
                    OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                    OUT[idx,2] = x[idx,2]^z
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                        OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                    OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                        OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                        col += Int32(1)
                    end
                end
            else
                if x[idx,3] == x[idx,4]
                    OUT[idx,1] = eps_min*eps_min^(z-1)
                    OUT[idx,2] = x[idx,2]^z
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = 0.0
                        OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = eps_min*eps_min^(z-1)
                    OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = 0.0
                        OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                        col += Int32(1)
                    end
                end
            end
        else
            if eps_min >= x[idx,1]
                if x[idx,3] == x[idx,4]
                    OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                    OUT[idx,2] = eps_max^z
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                        OUT[idx,end-1*colmax+col] = 0.0
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                    OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - eps_max) + x[idx,4]^z*(eps_max - x[idx,3]))/(x[idx,4] - x[idx,3])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                        OUT[idx,end-1*colmax+col] = 0.0
                        col += Int32(1)
                    end
                end
            elseif x[idx,2] >= eps_min
                if x[idx,3] == x[idx,4]
                    OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                    OUT[idx,2] = eps_max^z
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                        OUT[idx,end-1*colmax+col] = 0.0
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                    OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - eps_max) + x[idx,4]^z*(eps_max - x[idx,3]))/(x[idx,4] - x[idx,3])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                        OUT[idx,end-1*colmax+col] = 0.0
                        col += Int32(1)
                    end
                end
            else
                if x[idx,3] == x[idx,4]
                    OUT[idx,1] = eps_min*eps_min^(z-1)
                    OUT[idx,2] = eps_max^z
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = 0.0
                        OUT[idx,end-1*colmax+col] = 0.0
                        col += Int32(1)
                    end
                else
                    OUT[idx,1] = eps_min*eps_min^(z-1)
                    OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - eps_max) + x[idx,4]^z*(eps_max - x[idx,3]))/(x[idx,4] - x[idx,3])
                    while col <= colmax
                        OUT[idx,end-2*colmax+col] = 0.0
                        OUT[idx,end-1*colmax+col] = 0.0
                        col += Int32(1)
                    end
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

function SCMC_even_power_kernel_long(OUT::CuDeviceMatrix, x::CuDeviceMatrix, z::Integer)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    colmax = Int32((size(OUT,2)-4)/2)

    while idx <= Int32(size(OUT,1))
        # Reset the column counter
        col = Int32(1)

        if x[idx,4] <= 0.0
            OUT[idx,3] = x[idx,4]^z
            OUT[idx,4] = x[idx,3]^z
            if x[idx,2] >= x[idx,1]
                if x[idx,1] == x[idx,2]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = x[idx,1]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                elseif x[idx,1] >= x[idx,3]
                    if x[idx,1] >= x[idx,4]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                            OUT[idx,2] = x[idx,1]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    elseif x[idx,4] >= x[idx,2]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                            OUT[idx,2] = x[idx,1]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1] ) + x[idx,4]^z*(x[idx,1]  - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    else
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,4]*x[idx,4]^(z-1)
                            OUT[idx,2] = x[idx,1]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,4]*x[idx,4]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    end
                elseif x[idx,3] >= x[idx,2]
                    if x[idx,1] >= x[idx,4]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,1] *x[idx,1]^(z-1)
                            OUT[idx,2] = x[idx,2]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,1] *x[idx,1]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    elseif x[idx,4] >= x[idx,2]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                            OUT[idx,2] = x[idx,2]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    else
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,4]*x[idx,4]^(z-1)
                            OUT[idx,2] = x[idx,2]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,4]*x[idx,4]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    end
                else
                    if x[idx,1] >= x[idx,4]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                            OUT[idx,2] = x[idx,3]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,3]) + x[idx,4]^z*(x[idx,3] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        end
                    elseif x[idx,4] >= x[idx,2]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                            OUT[idx,2] = x[idx,3]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,3]) + x[idx,4]^z*(x[idx,3] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        end
                    else
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,4]*x[idx,4]^(z-1)
                            OUT[idx,2] = x[idx,3]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,4]*x[idx,4]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,3]) + x[idx,4]^z*(x[idx,3] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        end
                    end
                end
            elseif x[idx,3] >= x[idx,1]
                if x[idx,4] >= x[idx,1]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = x[idx,1]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                elseif x[idx,2] >= x[idx,4]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = x[idx,1]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                else
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,4]*x[idx,4]^(z-1)
                        OUT[idx,2] = x[idx,1]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,4]*x[idx,4]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                end
            elseif x[idx,2] >= x[idx,3]
                if x[idx,4] >= x[idx,1]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,1] *x[idx,1]^(z-1)
                        OUT[idx,2] = x[idx,2]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,1] *x[idx,1]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                elseif x[idx,2] >= x[idx,4]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = x[idx,2]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                else
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,4]*x[idx,4]^(z-1)
                        OUT[idx,2] = x[idx,2]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,4]*x[idx,4]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                end
            else
                if x[idx,4] >= x[idx,1]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = x[idx,3]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,3]) + x[idx,4]^z*(x[idx,3] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                elseif x[idx,2] >= x[idx,4]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = x[idx,3]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,3]) + x[idx,4]^z*(x[idx,3] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                else
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,4]*x[idx,4]^(z-1)
                        OUT[idx,2] = x[idx,3]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,4]*x[idx,4]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,3]) + x[idx,4]^z*(x[idx,3] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                end
            end
            
        elseif x[idx,3] >= 0.0
            OUT[idx,3] = x[idx,3]^z
            OUT[idx,4] = x[idx,4]^z
            if x[idx,2] >= x[idx,1]
                if x[idx,1] == x[idx,2]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = x[idx,1]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                elseif x[idx,1] >= x[idx,4]
                    if x[idx,1] >= x[idx,3]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                            OUT[idx,2] = x[idx,1]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    elseif x[idx,3] >= x[idx,2]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                            OUT[idx,2] = x[idx,1]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1] ) + x[idx,4]^z*(x[idx,1]  - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    else
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,3]*x[idx,3]^(z-1)
                            OUT[idx,2] = x[idx,1]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,3]*x[idx,3]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    end
                elseif x[idx,4] >= x[idx,2]
                    if x[idx,1] >= x[idx,3]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,1] *x[idx,1]^(z-1)
                            OUT[idx,2] = x[idx,2]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,1] *x[idx,1]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    elseif x[idx,3] >= x[idx,2]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                            OUT[idx,2] = x[idx,2]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    else
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,3]*x[idx,3]^(z-1)
                            OUT[idx,2] = x[idx,2]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,3]*x[idx,3]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    end
                else
                    if x[idx,1] >= x[idx,3]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                            OUT[idx,2] = x[idx,4]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,4]) + x[idx,4]^z*(x[idx,4] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        end
                    elseif x[idx,3] >= x[idx,2]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                            OUT[idx,2] = x[idx,4]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,4]) + x[idx,4]^z*(x[idx,4] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        end
                    else
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,3]*x[idx,3]^(z-1)
                            OUT[idx,2] = x[idx,4]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,3]*x[idx,3]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,4]) + x[idx,4]^z*(x[idx,4] - x[idx,3]))/(x[idx,4] - x[idx,3])
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
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = x[idx,1]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                elseif x[idx,2] >= x[idx,3]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = x[idx,1]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                else
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,3]*x[idx,3]^(z-1)
                        OUT[idx,2] = x[idx,1]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,3]*x[idx,3]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                end
            elseif x[idx,2] >= x[idx,4]
                if x[idx,3] >= x[idx,1]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,1] *x[idx,1]^(z-1)
                        OUT[idx,2] = x[idx,2]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,1] *x[idx,1]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                elseif x[idx,2] >= x[idx,3]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = x[idx,2]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                else
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,3]*x[idx,3]^(z-1)
                        OUT[idx,2] = x[idx,2]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,3]*x[idx,3]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                end
            else
                if x[idx,3] >= x[idx,1]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = x[idx,4]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,4]) + x[idx,4]^z*(x[idx,4] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                elseif x[idx,2] >= x[idx,3]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = x[idx,4]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,4]) + x[idx,4]^z*(x[idx,4] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                else
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,3]*x[idx,3]^(z-1)
                        OUT[idx,2] = x[idx,4]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,3]*x[idx,3]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,4]) + x[idx,4]^z*(x[idx,4] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                end
            end
        elseif abs(x[idx,3]) >= abs(x[idx,4])
            OUT[idx,3] = 0.0
            OUT[idx,4] = x[idx,3]^z
            if x[idx,2] >= x[idx,1]
                if x[idx,1] == x[idx,2]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = x[idx,1]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                elseif x[idx,1] >= x[idx,3]
                    if x[idx,1] >= 0.0
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                            OUT[idx,2] = x[idx,1]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    elseif 0.0 >= x[idx,2]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                            OUT[idx,2] = x[idx,1]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1] ) + x[idx,4]^z*(x[idx,1]  - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    else
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = 0.0
                            OUT[idx,2] = x[idx,1]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = 0.0
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    end
                elseif x[idx,3] >= x[idx,2]
                    if x[idx,1] >= 0.0
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,1] *x[idx,1]^(z-1)
                            OUT[idx,2] = x[idx,2]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,1] *x[idx,1]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    elseif 0.0 >= x[idx,2]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                            OUT[idx,2] = x[idx,2]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    else
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = 0.0
                            OUT[idx,2] = x[idx,2]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = 0.0
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    end
                else
                    if x[idx,1] >= 0.0
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                            OUT[idx,2] = x[idx,3]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,3]) + x[idx,4]^z*(x[idx,3] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        end
                    elseif 0.0 >= x[idx,2]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                            OUT[idx,2] = x[idx,3]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,3]) + x[idx,4]^z*(x[idx,3] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        end
                    else
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = 0.0
                            OUT[idx,2] = x[idx,3]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = 0.0
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,3]) + x[idx,4]^z*(x[idx,3] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        end
                    end
                end
            elseif x[idx,3] >= x[idx,1]
                if 0.0 >= x[idx,1]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = x[idx,1]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                elseif x[idx,2] >= 0.0
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = x[idx,1]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                else
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = 0.0
                        OUT[idx,2] = x[idx,1]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = 0.0
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                end
            elseif x[idx,2] >= x[idx,3]
                if 0.0 >= x[idx,1]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,1] *x[idx,1]^(z-1)
                        OUT[idx,2] = x[idx,2]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,1] *x[idx,1]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                elseif x[idx,2] >= 0.0
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = x[idx,2]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                else
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = 0.0
                        OUT[idx,2] = x[idx,2]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = 0.0
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                end
            else
                if 0.0 >= x[idx,1]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = x[idx,3]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,3]) + x[idx,4]^z*(x[idx,3] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                elseif x[idx,2] >= 0.0
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = x[idx,3]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,3]) + x[idx,4]^z*(x[idx,3] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                else
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = 0.0
                        OUT[idx,2] = x[idx,3]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = 0.0
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,3]) + x[idx,4]^z*(x[idx,3] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                end
            end
        else
            OUT[idx,3] = 0.0
            OUT[idx,4] = x[idx,4]^z
            if x[idx,2] >= x[idx,1]
                if x[idx,1] == x[idx,2]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = x[idx,1]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                elseif x[idx,1] >= x[idx,4]
                    if x[idx,1] >= 0.0
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                            OUT[idx,2] = x[idx,1]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    elseif 0.0 >= x[idx,2]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                            OUT[idx,2] = x[idx,1]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1] ) + x[idx,4]^z*(x[idx,1]  - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    else
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = 0.0
                            OUT[idx,2] = x[idx,1]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = 0.0
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    end
                elseif x[idx,4] >= x[idx,2]
                    if x[idx,1] >= 0.0
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,1] *x[idx,1]^(z-1)
                            OUT[idx,2] = x[idx,2]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,1] *x[idx,1]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    elseif 0.0 >= x[idx,2]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                            OUT[idx,2] = x[idx,2]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    else
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = 0.0
                            OUT[idx,2] = x[idx,2]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = 0.0
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                                col += Int32(1)
                            end
                        end
                    end
                else
                    if x[idx,1] >= 0.0
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                            OUT[idx,2] = x[idx,4]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,4]) + x[idx,4]^z*(x[idx,4] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        end
                    elseif 0.0 >= x[idx,2]
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                            OUT[idx,2] = x[idx,4]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,4]) + x[idx,4]^z*(x[idx,4] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        end
                    else
                        if x[idx,3] == x[idx,4]
                            OUT[idx,1] = 0.0
                            OUT[idx,2] = x[idx,4]^z
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        else
                            OUT[idx,1] = 0.0
                            OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,4]) + x[idx,4]^z*(x[idx,4] - x[idx,3]))/(x[idx,4] - x[idx,3])
                            while col <= colmax
                                OUT[idx,end-2*colmax+col] = 0.0
                                OUT[idx,end-1*colmax+col] = 0.0
                                col += Int32(1)
                            end
                        end
                    end
                end
            elseif x[idx,4] >= x[idx,1]
                if 0.0 >= x[idx,1]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = x[idx,1]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                elseif x[idx,2] >= 0.0
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = x[idx,1]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                else
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = 0.0
                        OUT[idx,2] = x[idx,1]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = 0.0
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,1]) + x[idx,4]^z*(x[idx,1] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-2*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                end
            elseif x[idx,2] >= x[idx,4]
                if 0.0 >= x[idx,1]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,1] *x[idx,1]^(z-1)
                        OUT[idx,2] = x[idx,2]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,1] *x[idx,1]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                elseif x[idx,2] >= 0.0
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = x[idx,2]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                else
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = 0.0
                        OUT[idx,2] = x[idx,2]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = 0.0
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,2]) + x[idx,4]^z*(x[idx,2] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = x[idx,end-1*colmax+col] * (x[idx,4]^z - x[idx,3]^z)/(x[idx,4] - x[idx,3])
                            col += Int32(1)
                        end
                    end
                end
            else
                if 0.0 >= x[idx,1]
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = x[idx,4]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,1]*x[idx,1]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,4]) + x[idx,4]^z*(x[idx,4] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-2*colmax+col] * z*x[idx,1]^(z-1)
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                elseif x[idx,2] >= 0.0
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = x[idx,4]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = x[idx,2]*x[idx,2]^(z-1)
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,4]) + x[idx,4]^z*(x[idx,4] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = x[idx,end-1*colmax+col] * z*x[idx,2]^(z-1)
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
                else
                    if x[idx,3] == x[idx,4]
                        OUT[idx,1] = 0.0
                        OUT[idx,2] = x[idx,4]^z
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    else
                        OUT[idx,1] = 0.0
                        OUT[idx,2] = (x[idx,3]^z*(x[idx,4] - x[idx,4]) + x[idx,4]^z*(x[idx,4] - x[idx,3]))/(x[idx,4] - x[idx,3])
                        while col <= colmax
                            OUT[idx,end-2*colmax+col] = 0.0
                            OUT[idx,end-1*colmax+col] = 0.0
                            col += Int32(1)
                        end
                    end
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