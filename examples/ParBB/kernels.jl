
# First one, we have a matrix of booleans. We want to identify the first element
# in each row that's true (1), and return that as an array.
function first_true_kernel(bool_matrix, first_true_indices)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    matsize = Int32(size(bool_matrix, 2))
    while idx <= Int32(length(first_true_indices))
        first_true_indices[idx] = Int32(1)
        while first_true_indices[idx] <= matsize
            if bool_matrix[idx, first_true_indices[idx]]
                first_true_indices[idx] += matsize+Int32(1)
            else
                first_true_indices[idx] += Int32(1)
            end
        end
        first_true_indices[idx] -= (matsize+Int32(1))
        idx += stride
    end
    return nothing
end


# This one accesses elements in the matrix
function access_kernel(matrix, n_rows, pivot_cols, pivot_col_vals)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    while idx <= Int32(length(pivot_col_vals))
        col_index = cld(idx,n_rows)
        if pivot_cols[col_index] != Int32(0)
            pivot_col_vals[idx] = matrix[idx, pivot_cols[cld(idx,n_rows)]]
        else
            pivot_col_vals[idx] = -Inf
        end
        idx += stride
    end
    return nothing
end

# These set specific values to Inf, zero, or one, respectively
function set_inf_kernel(vec, inds)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    while idx <= Int32(length(inds))
        if inds[idx]
            vec[idx] = Inf
        end
        idx += stride
    end
    return nothing
end


function set_zero_kernel(vec, inds)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    while idx <= Int32(length(inds))
        if inds[idx]
            vec[idx] = 0.0
        end
        idx += stride
    end
    return nothing
end
function set_one_kernel(vec, inds)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    while idx <= Int32(length(inds))
        if inds[idx]
            vec[idx] = 1.0
        end
        idx += stride
    end
    return nothing
end

# # Sometimes multiplication and division ends up with a value
# # close to 0, but not exactly 0. Sometimes, these can
# # mess up pivots by causing some values to get multiplied
# # by 1/[very low value], so there are some 1E16's floating
# # around, and this can sometimes cause pivots to fail.
# function filter_kernel(mat)
#     idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
#     stride = blockDim().x * gridDim().x
#     len = Int32(size(mat, 1))
#     wid = Int32(size(mat, 2))
#     tot = len*wid
#     while idx <= tot
#         if abs(mat[idx])<=1E-15
#             mat[idx] = 0.0
#         end
#         idx += stride
#     end
#     return nothing
# end

# This one finds the index of the minimum value of each column
# of a ratio vector
function find_min_kernel(ratios, n_rows, rows)
    row = blockIdx().x
    col = threadIdx().x

    # Need to use parallel reduction... Each block handles one row...
    shared_min_vals = @cuDynamicSharedMem(Float64, blockDim().x)
    shared_min_idxs = @cuDynamicSharedMem(Int32, blockDim().x, offset=Int32(8)*blockDim().x)

    # Note: "row" and "col" are a little weird here, because
    # "ratios" is actually a vector.

    # Initialize the shared memory
    shared_min_vals[col] = ratios[(row-1)*n_rows + col]
    shared_min_idxs[col] = col

    # Parallel reduction to find the index of the minimum value
    # (This could probably be made more efficient. If n_rows is
    #  not cleanly divisble by 2 there may be repeated checks
    #  of some elements, plus at smaller than 32 elements, not all
    #  threads in each warp will be used.)
    stride = cld(n_rows,Int32(2))
    while (stride != Int32(0))
        sync_threads()
        if (col <= stride) && (col+stride <= n_rows)
            if shared_min_vals[col] == shared_min_vals[col + stride]
                if shared_min_idxs[col] > shared_min_idxs[col + stride]
                    shared_min_idxs[col] = shared_min_idxs[col + stride]
                end
            elseif shared_min_vals[col] > shared_min_vals[col + stride]
                shared_min_vals[col] = shared_min_vals[col + stride]
                shared_min_idxs[col] = shared_min_idxs[col + stride]
            end
        end 
        if stride==1
            stride = Int32(0)
        else
            stride = cld(stride,Int32(2))
        end
    end
    if col==1
        rows[row] = shared_min_idxs[1]
    end
    return nothing
end

# This kernel function performs the pivoting operation
# based on the previously determined pivot rows and columns.
# Each block takes on a single LP to pivot, so the number
# of blocks should match n_probs, threads can be anything but
# 32 should be fine, shmem should be 8*(size(tableau, 2)+n_rows)
function pivot_kernel(tableau, pivot_rows, pivot_cols, n_rows, width, entries_per_LP)
    # Since each block has its own problem to address, we
    # need to figure out the correct row/col addresses based
    # on block.
    thread = threadIdx().x
    LP = (blockIdx().x)-Int32(1) #Base 0 is easier in some cases

    # Maybe let's start with setting up shared memory. We only
    # need the pivot row and the factors
    pivot_row = @cuDynamicSharedMem(Float64, width)
    factors = @cuDynamicSharedMem(Float64, n_rows, offset=Int32(8*width))

    # Before we do anything at all, skip the entire pivot operation
    # if pivot_cols[LP] is 0.
    if pivot_cols[blockIdx().x] != 0

        # Fill in the pivot row with the correct pivot rows
        stride = blockDim().x 
        while thread <= width
            pivot_row[thread] = tableau[LP*n_rows + pivot_rows[blockIdx().x], thread]
            thread += stride
        end
        sync_threads()
        thread = threadIdx().x

        # Fill in the multiplication factors for each row by comparing against 
        # the pivot element in the pivot column
        while thread <= n_rows
            if thread==pivot_rows[blockIdx().x]
                factors[thread] = 1/tableau[LP*n_rows + thread, pivot_cols[blockIdx().x]] - 1
            else
                factors[thread] = -tableau[LP*n_rows + thread, pivot_cols[blockIdx().x]] / pivot_row[pivot_cols[blockIdx().x]]
            end
            thread += stride
        end
        sync_threads()
        thread = threadIdx().x

        # Now we've stored the pivot row and multiplicative factors separately 
        # from the main tableau, the pivot can happen by referencing these saved
        # values.
        while thread <= entries_per_LP
            LP_row = cld(thread,width) #We need both division and modulo, but we can calculate one from the other
            # Note: We could also have each thread take one column in its LP to not use division, 
            # but that might not be very efficient if the number of columns isn't neatly divisible
            # by 32.
            row = Int32(LP*n_rows) + Int32(LP_row)
            col = thread - Int32(((LP_row-Int32(1))*width))

            # Now that we know what row and column we're looking at, we can
            # perform pivots.
            # Deal with the pivot column specially, but for all others, we're fine.
            if col==pivot_cols[blockIdx().x]
                if LP_row==pivot_rows[blockIdx().x]
                    tableau[row,col] = 1.0
                else
                    tableau[row,col] = 0.0
                end
            else
                tableau[row,col] += factors[LP_row]*pivot_row[col]
            end
            thread += stride
        end
    end
    sync_threads()
    return nothing
end


# This kernel checks a Boolean matrix "mat" and sets output to "true" if
# every element of mat is "false". If any element of mat is "true",
# then output is "false".
function not_any_kernel(output, mat)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    total_size = Int32(size(mat, 1))*Int32(size(mat,2))

    while output[1] && idx <= total_size 
        if mat[idx]
            output[1] = false
        end
        idx += stride
    end

    return nothing
end

# This kernel checks a Boolean matrix "mat" and sets the output to "true"
# if every element of mat is "Inf".
function all_inf_kernel(output, mat)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    len = Int32(length(mat))

    while output[1]==true && idx <= len
        if ~isinf(mat[idx])
            output[1] = false
        end
        idx += stride
    end
    return nothing
end

# This kernel checks if rows of the two matrices are equal.
# If they're completely equal, we mark the flag as true
# for that row.
function degeneracy_check_kernel(flag, mat1, mat2)
    row = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    len = Int32(length(flag))
    width = Int32(size(mat1, 2))

    # Check each row of the matrix to see if the flag should be
    # true or false
    while row <= len
        col = Int32(1)
        while col <= width
            if abs(mat1[row,col]-mat2[row,col]) <= 1E-10
                if col==width
                    @inbounds flag[row] = true
                end
            else
                @inbounds flag[row] = false
                break
            end
            col += Int32(1)
        end
        row += stride
    end
    return nothing
end

# This kernel creates the basic tableau skeleton, which involves placing
# 1's and -1's in the correct spaces based on the number of variables. 
# An example tableau (with labels) might look like the following:
#=
OLD TABLEAU:
9×19 CuArray{Float64, 2, CUDA.DeviceMemory}:
   Z'   X_1        X_2        X_3        S_1  S_2  S_3  S_4  S_5  S_6  S_7  A_1  A_2  A_3  A_4  A_5  A_6  A_7  B
  1.0   0.0        0.0        0.0        1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.668527
  0.0   1.0        0.0        0.0        0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0
  0.0   0.0        1.0        0.0        0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0
  0.0   0.0        0.0        1.0        0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0
  1.0  -0.317476  -0.449098  -0.330095   0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
  0.0   0.0        0.0        0.0        0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
  0.0   0.0        0.0        0.0        0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 -1.0   0.0        0.0        0.0        0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
  0.0   0.0        0.0        0.0        0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0

Description: (OLD)
 Z is the epigraph variable that replaces the objective function. From the 
    lower bound of the relaxation of the objective function, we calculate
    that Z >= -0.668527. Adding in a slack variable, we get Z - S_1 = -0.668527. 
    But, this is inconvenient since the slack variable is negative, so we can 
    negate the expression above and instead calculate Z', where Z' = -Z. 
    Now we have Z' <= 0.668527, or Z' + S_1 = 0.668527. This is the first row
    of the tableau. 
Since we originally wanted to minimize Z, but have replaced Z by Z', we are 
    now minimizing -Z'. This is the second to last row of the tableau.
X_1 through X_3 are the problem variables, scaled to be on the domain [0,1].
    Since the tableau method assumes all variables are >= 0, we only need to
    include rows for the upper bounds. I.e., X_1 <= 1.0. Adding in slack variables,
    we get X_1 + S_2 = 1.0 (and similar for X_2 and X_3). This is lines 2-4
    of the tableau.
Line 5 of the tableau would not be created in this step, but shows an example
    of what the line will eventually contain. Here, it is a convex relaxation
    of the objective function, hence the 1 in the Z' column, and the subgradient
    information is stored in the columns for the X variables. This line can be
    obtained in the following way:
    Z >= -0.317476*X_1 + -0.449098*X_2 + -0.330095*X_3
    -0.317476*X_1 + -0.449098*X_2 + -0.330095*X_3 <= Z
    -0.317476*X_1 + -0.449098*X_2 + -0.330095*X_3 - Z <= 0
    -0.317476*X_1 + -0.449098*X_2 + -0.330095*X_3 + Z' <= 0
    -0.317476*X_1 + -0.449098*X_2 + -0.330095*X_3 + Z' + S_5 = 0
Lines 6 and 7 are reserved for future cuts. There may be greater or fewer lines
    dedicated to cuts for the objective function and/or constraints. In the pivoting
    step, these unused rows will be ignored until they contain useful information.
In this case, none of the slack variables were negative, so there was no need
    to use the artificial variable columns (A_1 through A_7). This information
    will be passed to the pivoting kernel, which will ignore these columns entirely.
The final row is for the Phase I objective, which is to minimize the sum of the 
    artificial variables. This is unnecessary in this case, as no Phase I solution
    is necessary and we already have a BFS comprising S_1 through S_5 basic.

NEW TABLEAU:
9×17 CuArray{Float64, 2, CUDA.DeviceMemory}:
  Z_+  Z_-   X_1        X_2        X_3        S_1  S_2  S_3  S_4  S_5  S_6  S_7  A_1  A_2  A_3  A_4  B
  0.0  0.0   1.0        0.0        0.0        1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0      # Variable 1 row
  0.0  0.0   0.0        1.0        0.0        0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0      # Variable 2 row
  0.0  0.0   0.0        0.0        1.0        0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0      # Variable 3 row
 -1.0  1.0   0.0        0.0        0.0        0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.668527 # Lower bound cut
  1.0 -1.0   0.317476   0.449098   0.330095   0.0  0.0  0.0  0.0 -1.0  0.0  0.0  0.0  1.0  0.0  0.0  0.2      # First cut
  0.0  0.0   0.0        0.0        0.0        0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0      # Empty for second cut
  0.0  0.0   0.0        0.0        0.0        0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0      # Empty for third cut
  1.0 -1.0   0.0        0.0        0.0        0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0      # Phase II objective
 -1.0  1.0  -0.317476  -0.449098  -0.330095   0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0 -0.2      # Phase I objective

Note that the lower bound cut comes from something like the following:
 [objective]     >= -0.668527
 Z_+ - Z_-       >= -0.668527
 Z_+ - Z_- - S_4 == -0.668527
-Z_+ + Z_- + S_4 == 0.668527

And for objective function cuts, we have something like the following:
            -0.317476*X_1 - 0.449098*X_2 - 0.330095*X_3 + 0.2       <= (Z_+ - Z_-)
-Z_+ + Z_- - 0.317476*X_1 - 0.449098*X_2 - 0.330095*X_3             <= -0.2
-Z_+ + Z_- - 0.317476*X_1 - 0.449098*X_2 - 0.330095*X_3 + S_5       == -0.2
 Z_+ - Z_- + 0.317476*X_1 + 0.449098*X_2 + 0.330095*X_3 - S_5 + A_2 == 0.2

 With the negative of the whole row, not including the artificial variable, being added to the
 phase I objective row. 
=#
function tableau_skeleton_kernel(tableau, n_vars, slack_count)
    # Each block handles one LP
    thread = threadIdx().x
    LP = blockIdx().x #Base 1 is easier in this kernel
    stride = blockDim().x
    width = n_vars*Int32(2) + Int32(3) # Skipping unnecessary slack variables and artificial variables
    entries_per_LP = width*(n_vars+Int32(1))
    
    while thread <= entries_per_LP
        LP_row = cld(thread,width) #We need both division and modulo, but we can calculate one from the other
        row = (LP-Int32(1))*(slack_count+Int32(2)) + Int32(LP_row) # This is the real row in the tableau
        col = thread - Int32(((LP_row-Int32(1))*width))

        # Fill in variable columns (leaving first 2 columns for epigraph variables)
        if (LP_row == col) && (LP_row <= n_vars)
            tableau[row, col + Int32(2)] = 1.0

        # Fill in slack variables (leaving first 2 columns for epigraph variables)
        elseif ((LP_row + n_vars) == col) && (LP_row <= n_vars)
            tableau[row, col + Int32(2)] = 1.0

        # Fill in the b column with variable upper bounds (all scaled to 1.0)
        elseif (col == width) && (LP_row <= n_vars)
            tableau[row, end] = 1.0

        # Fill in the phase II objective row
        elseif (LP_row == (n_vars + Int32(1))) && (col == Int32(1))
            tableau[LP*(slack_count+Int32(2)) - Int32(1), col] = 1.0
        elseif (LP_row == (n_vars + Int32(1))) && (col == Int32(2))
            tableau[LP*(slack_count+Int32(2)) - Int32(1), col] = -1.0
        end

        thread += stride
    end
    sync_threads()
    return nothing
end


# Make sure to call this with the number of threads being
# the largest value of 2^n that's less than the number of columns
function accumulate_mul_kernel(output, mat1, mat2)
    row = blockIdx().x
    col = threadIdx().x
    n_cols = Int32(size(mat1, 2))

    # Each block handles one row
    shared_mults = @cuDynamicSharedMem(Float64, n_cols)

    # Initialize the shared memory to be the element-wise multiplication
    # of mat1 and mat2
    stride = blockDim().x # e.g., 32, where there are 40 columns
    while col <= n_cols
        shared_mults[col] = mat1[row,col] * mat2[row,col]
        col += stride
    end
    sync_threads()

    # Now perform a parallel reduction to sum up each row
    col = threadIdx().x
    while stride > 0
        if (col <= stride) && (col+stride <= n_cols)
            shared_mults[col] += shared_mults[col+stride]
        end
        stride >>= 1 # Bitshift by 1 to divide by 2
        sync_threads()
    end

    if col==1
        output[row] = shared_mults[1]
    end
    sync_threads()
    return nothing
end

# This kernel adds information for a cut to the tableau. It modifies
# only the row where the cut is being placed, and the final row, which
# contains the Phase I objective. "obj_flag" should be "true" if the
# cut is for the problem objective function, and "false" for a standard
# constraint. "geq_flag" should be true if the constraint is a GEQ
# constraint.
function add_cut_kernel(tableau, subgradients, final_col, active_row, n_vars, 
                        n_rows, n_probs, slack_count, obj_flag, geq_flag) #, deg_flag)
    # For each LP, we modify (active_row) and (n_rows), being the line
    # where the cut is added and the phase I objective row. Each thread
    # modifies a single LP, since it's just filling in values in one line,
    # the number of additions is small, and the points to fill in are sporadic.
    thread = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x

    while thread <= n_probs
        # if deg_flag[thread]
        #     thread += stride
        #     continue
        # end
        row = (thread - Int32(1))*n_rows + active_row
        if final_col[thread] < 0
            if geq_flag==true 
                # This is a GEQ constraint, so the slack variable would be negative,
                # but the final column is also negative. This means we negate the
                # whole row, so both the slack and final column are positive. No
                # need for an artificial variable.

                # Add in the epigraph variables for the objective (NEGATED)
                if obj_flag
                    tableau[row, Int32(1)] = 1.0 # Negative of the normal -1
                    tableau[row, Int32(2)] = -1.0 # Negative of the normal +1
                end
                
                # Add the subgradients for the cut (NEGATED)
                col = Int32(3)
                while col <= n_vars + Int32(2)
                    tableau[row, col] = -subgradients[thread, col-Int32(2)]
                    col += Int32(1)
                end

                # Add the slack variable (DOUBLE NEGATED)
                tableau[row, Int32(2) + n_vars + active_row] = 1.0

                # Add the final column value (NEGATED)
                tableau[row, end] = -final_col[thread]
            else
                # This is a LEQ constraint, so the slack variable would be positive,
                # but the final column is negative. This means we negate the whole
                # row, add an artificial variable, and add the original row to the 
                # phase I objective
                art_row = thread*n_rows

                # Add in the epigraph variables (NEGATED), and for the artificial objective (DOUBLE NEGATED)
                if obj_flag
                    tableau[row, Int32(1)] = 1.0 # Negative of the normal -1
                    tableau[row, Int32(2)] = -1.0 # Negative of the normal +1
                    tableau[art_row, Int32(1)] += -1.0
                    tableau[art_row, Int32(2)] += 1.0
                end

                # Add the subgradients for the cut (NEGATED), and for the artificial objective (DOUBLE NEGATED)
                col = Int32(3)
                while col <= n_vars + Int32(2)
                    tableau[row, col] = -subgradients[thread, col-Int32(2)]
                    tableau[art_row, col] += subgradients[thread, col-Int32(2)]
                    col += Int32(1)
                end

                # Add the slack variable (NEGATED) and artificial variable
                tableau[row, Int32(2) + n_vars + active_row] = -1.0
                tableau[art_row, Int32(2) + n_vars + active_row] = 1.0
                tableau[row, Int32(2) + slack_count + active_row] = 1.0

                # Add the final column value (NEGATED), and for the artificial objective (DOUBLE NEGATED)
                tableau[row, end] = -final_col[thread]
                tableau[art_row, end] += final_col[thread]
            end
        else # Final column would be positive
            if geq_flag==true 
                # This is a GEQ constraint, so the slack variable would be negative,
                # and the final column is positive. This means we keep the row as-is,
                # but we add an artificial variable
                art_row = thread*n_rows

                # Add in the epigraph variables for the objective
                if obj_flag
                    tableau[row, Int32(1)] = -1.0 
                    tableau[row, Int32(2)] = 1.0 
                    tableau[art_row, Int32(1)] += 1.0
                    tableau[art_row, Int32(2)] += -1.0
                end

                # Add the subgradients for the cut, and for the artificial objective (NEGATED)
                col = Int32(3)
                while col <= n_vars + Int32(2)
                    tableau[row, col] = subgradients[thread, col-Int32(2)]
                    tableau[art_row, col] += -subgradients[thread, col-Int32(2)]
                    col += Int32(1)
                end

                # Add the slack variable (NEGATED) and artificial variable
                tableau[row, Int32(2) + n_vars + active_row] = -1.0
                tableau[art_row, Int32(2) + n_vars + active_row] = 1.0
                tableau[row, Int32(2) + slack_count + active_row] = 1.0

                # Add the final column value, and for the artificial objective (NEGATED)
                tableau[row, end] = final_col[thread]
                tableau[art_row, end] += -final_col[thread]
            else
                # This is a LEQ constraint, and the final column is positive.
                # We can add the subgradients as-is, and we don't need an artificial
                # variable.

                # Add in the epigraph variables for the objective
                if obj_flag
                    tableau[row, Int32(1)] = -1.0 
                    tableau[row, Int32(2)] = 1.0
                end
                
                # Add the subgradients for the cut
                col = Int32(3)
                while col <= n_vars + Int32(2)
                    tableau[row, col] = subgradients[thread, col-Int32(2)]
                    col += Int32(1)
                end

                # Add the slack variable
                tableau[row, Int32(2) + n_vars + active_row] = 1.0

                # Add the final column value
                tableau[row, end] = final_col[thread]
            end
        end
        thread += stride
    end
    sync_threads()
    return nothing
end
  
# A simplified version of add_cut_kernel that only adds the objective
# function lower bound to the tableau. Subgradients are not required,
# because an epigraph reformulation is used and the subgradients on
# this line are always 0.
function add_lower_bound_kernel(tableau, final_col, active_row, n_rows, n_probs)
    # For each LP, we modify (active_row) and (n_rows), being the line
    # where the "cut" is added and the phase I objective row. Each thread
    # modifies a single LP, since it's just filling in values in one line,
    # the number of additions is small, and the points to fill in are sporadic.
    thread = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x

    while thread <= n_probs
        row = (thread - Int32(1))*n_rows + active_row
        # The lower bound constraint is a GEQ constraint, so a positive value
        # in the final column means the slack variable will have a negative
        # coefficient. This means we'll need an artificial variable, and must
        # add the negative of the row to the phase I objective row.
        if final_col[thread] > 0
            art_row = thread*n_rows
            tableau[row, Int32(1)] = 1.0 # Z_+, negated from -1
            tableau[row, Int32(2)] = -1.0 # Z_-, negated from 1
            tableau[row, Int32(2)*active_row + Int32(1)] = -1.0 # Slack variable
            tableau[row, active_row + n_rows] = 1.0 # Artificial variable
            tableau[row, end] = final_col[thread] # b

            tableau[art_row, Int32(1)] = -1.0
            tableau[art_row, Int32(2)] = 1.0
            tableau[art_row, Int32(2)*active_row + Int32(1)] = 1.0
            tableau[art_row, end] = -final_col[thread]
        
        # In the other case, the lower bound is nonpositive. Since it's a GEQ
        # constraint, the slack variable starts as negative (or 0), but we can
        # negate the whole row to make the slack variable and final column 
        # nonnegative.
        else
            tableau[row, Int32(1)] = -1.0
            tableau[row, Int32(2)] = 1.0
            tableau[row, Int32(2)*active_row + Int32(1)] = 1.0 # Slack variable
            tableau[row, end] = -final_col[thread]
        end
        thread += stride
    end
    sync_threads()
    return nothing
end

# This kernel transitions from Phase I to Phase II. It checks the 
# status of the Phase I objective and marks the Phase II solution
# as -Inf if Phase I did not find a feasible solution.
function feasibility_check_kernel(tableau, n_rows, n_probs)
    thread = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    width = Int32(size(tableau, 2))

    # Each thread will check one LP and then move on
    while thread <= n_probs
        # Check if a BFS was NOT found.
        if (tableau[thread*n_rows, end] < -1E-14) || (tableau[thread*n_rows, end] > 1E-14)
            # The thread will adjust all columns of the preceding row.
            # This isn't very efficient, but if the width of the tableau
            # is much less than the number of problems, this may be
            # more efficient than setting each block to check one LP.
            # Also, this will be very quick if every LP found a BFS.
            col = Int32(1)
            while col < width
                tableau[thread*n_rows-Int32(1), col] = 0.0
                col += Int32(1)
            end
            tableau[thread*n_rows-Int32(1), end] = -Inf
        end

        # Regardless of whether a BFS was found or not, set the phase I
        # objective row to all 0s
        col = Int32(1)
        while col <= width
            tableau[thread*n_rows, col] = 0.0
            col += Int32(1)
        end
        thread += stride
    end
    sync_threads()
    return nothing
end
#=
tableau = [1.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  -1.0;
            0.0  1.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   1.0;
            0.0  0.0  1.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   1.0;
            0.0  0.0  0.0  1.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   1.0;
            0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0;
            0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0;
            0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0;
            -1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0;
            0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0;
            1.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  -2.0;
            0.0  1.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   1.0;
            0.0  0.0  1.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   1.0;
            0.0  0.0  0.0  1.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   1.0;
            0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0;
            0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0;
            0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0;
            -1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0;
            0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0   0.0];
subgradients = CuArray([-0.3176476 -0.449098 -0.330095; 0.32 0.45 0.33])
final_col = CuArray([5.0 -4.9])

@device_code_warntype @cuda threads=768 add_cut_kernel(tableau, subgradients, final_col, Int32(7), Int32(3), Int32(9), Int32(2), Int32(7))




function test_kernel(storage, rand_mat1, rand_mat2)
    @cuda blocks=1000 threads=1024 shmem=8*120 accumulate_mul_kernel(storage, rand_mat1, rand_mat2)
    return storage
end

rand_mat1 = CUDA.rand(Float64, 1000, 100);
rand_mat2 = CUDA.rand(Float64, 1000, 100);
storage = CuArray{Float64}(undef, 1000);
@device_code_warntype @cuda blocks=1000 threads=1024 shmem=8*120 accumulate_mul_kernel(storage, rand_mat1, rand_mat2)
tempstore = similar(rand_mat1);
CUDA.@time test_other(storage, rand_mat1, rand_mat2, tempstore);
function test_other(storage, rand_mat1, rand_mat2, tempstore)
    tempstore .= rand_mat1 .* rand_mat2
    storage .= sum(tempstore, dims=2)
    return storage
end


function SCRAP_find_min_kernel(ratios, n_rows, rows)
    row = blockIdx().x
    col = threadIdx().x

    # Need to use parallel reduction... Each block handles one row...
    shared_min_vals = @cuDynamicSharedMem(Float64, blockDim().x)
    shared_min_idxs = @cuDynamicSharedMem(Int32, blockDim().x, offset=Int32(8)*blockDim().x)

    # Note: "row" and "col" are a little weird here, because
    # "ratios" is actually a vector.

    # Initialize the shared memory
    shared_min_vals[col] = ratios[(row-1)*n_rows + col]
    shared_min_idxs[col] = col

    # Parallel reduction to find the index of the minimum value
    # (This could probably be made more efficient. If n_rows is
    #  not cleanly divisble by 2 there may be repeated checks
    #  of some elements, plus at smaller than 32 elements, not all
    #  threads in each warp will be used.)
    stride = cld(n_rows,Int32(2))
    while (stride != Int32(0))
        sync_threads()
        if (col <= stride) && (col+stride <= n_rows)
            if shared_min_vals[col] == shared_min_vals[col + stride]
                if shared_min_idxs[col] > shared_min_idxs[col + stride]
                    shared_min_idxs[col] = shared_min_idxs[col + stride]
                end
            elseif shared_min_vals[col] > shared_min_vals[col + stride]
                shared_min_vals[col] = shared_min_vals[col + stride]
                shared_min_idxs[col] = shared_min_idxs[col + stride]
            end
        end 
        if stride==1
            stride = Int32(0)
        else
            stride = cld(stride,Int32(2))
        end
    end
    if col==1
        rows[row] = shared_min_idxs[1]
    end
    return nothing
end


function DEPRECATED_create_tableau_kernel(tableau, n_vars, n_rows, n_cols, entries_per_LP, 
                                cuts, max_cuts, slack_count, corrected_subgradients, 
                                lower_bounds, b_vals)
    # We'll make each block handle a single LP.
    thread = threadIdx().x
    LP = (blockIdx().x)-Int32(1) #Base 0 is easier in some cases
    stride = blockDim().x 
    
    while thread <= entries_per_LP
        LP_row = cld(thread,width) #We need both division and modulo, but we can calculate one from the other
        # Note: We could also have each thread take one column in its LP to not use division, 
        # but that might not be very efficient if the number of columns isn't neatly divisible
        # by 32.
        row = Int32(LP*n_rows) + Int32(LP_row)
        col = thread - Int32(((LP_row-Int32(1))*width))

        # We now know what row and column we're looking at. We can fill in values as needed. 

        # Fill in the first column
        if col==1 # Lots of things happen for column 1
            if row <= cuts+1 #Lower bound constraint and each cut
                tableau[row, col] = 1.0
            elseif row == slack_count + Int32(1) # Objective row
                tableau[row, col] = -1.0
            end

        # Now fill in the row for the most recent cut
        elseif (row == cuts+1) && (col > 1) && (col <= n_vars + Int32(1))
            tableau[row, col] = corrected_subgradients[LP, col-Int32(1)]

        # Now fill in the 1's for slack variables
        elseif (row <= slack_count) && (row==(col + n_vars + Int32(1)))
            # Should add a skip for rows belonging to future cuts...
            tableau[row, col] = 1.0

        # Now fill in the lower bound for the epigraph variable
        elseif col==n_cols
            if row==1
                tableau[row, col] = -lower_bounds[LP]
        
        # Now the value for the first cut, from the b-value
            elseif row==2
                tableau[row, col] = b_vals[LP]
        
        # And finally, the upper bounds for all the variables
            elseif (row > max_cuts + Int32(1)) && (row <= max_cuts + Int32(1) + n_vars)
                tableau[row, col] = 1.0
            end

        # And now we change the variable upper bounds to be 1's
        elseif (row > max_cuts + Int32(1)) && (row <= max_cuts + Int32(1) + n_vars) && (row - max_cuts == col)
            tableau[row, col] = 1.0
        end

        thread += stride
    end
    sync_threads()
    return nothing
end

function DEPRECATED_tableau_skeleton_kernel(tableau, n_vars, slack_count, lower_bounds)
    # Each block handles one LP
    thread = threadIdx().x
    LP = (blockIdx().x)-Int32(1) #Base 0 is easier in some cases
    stride = blockDim().x
    width = ((n_vars + Int32(1))*2 + Int32(1))
    height = (n_vars + Int32(2))
    entries_per_LP = width*height
    
    while thread <= entries_per_LP
        LP_row = cld(thread,width) #We need both division and modulo, but we can calculate one from the other
        row = Int32(LP*(slack_count+Int32(2))) + Int32(LP_row) # This is the real row in the tableau
        col = thread - Int32(((LP_row-Int32(1))*width))

        # NOTE: No longer going to do this. We aren't doing an epigraph reformulation
        # anymore, so it's probably better to put in this row as a GEQ [lower bound]
        # "cut".
        # Fill in the objective lower bound constraint. Mark it as negative if
        # the associated lower bound is positive.
        if (col==1) && (LP_row==1)
            if lower_bounds[LP+Int32(1)] > 0.0
                tableau[row, col] = -1.0
            else
                tableau[row, col] = 1.0
            end

        # Fill in the variable upper bound rows (note that because this is "elseif",
        # we don't overwrite [1,1])
        elseif (col < height) && (LP_row==col)
            tableau[row, col] = 1.0
        
        # Fill in the slack variables for the lower bound row
        elseif (col == height) && (LP_row == Int32(1))
            if lower_bounds[LP+Int32(1)] > 0.0
                tableau[row, col] = -1.0
            else
                tableau[row, col] = 1.0
            end
        
        # Fill in slack variables for other rows
        elseif (col > height) && (col < width) && ((LP_row + n_vars + Int32(1)) == col)
            tableau[row, col] = 1.0

        # Fill in the lower bound value
        elseif (col == width) && (LP_row == Int32(1))
            if lower_bounds[LP+Int32(1)] > 0.0
                tableau[row, end] = lower_bounds[LP+Int32(1)]
            else
                tableau[row, end] = -lower_bounds[LP+Int32(1)]
            end

        # Fill in the upper bound values for the variables
        elseif (col == width) && (LP_row > Int32(1)) && (LP_row < height)
            tableau[row, end] = 1.0
        
        # Fill in the objective function row (one row before the end, because of the Phase I objective row)
        elseif (col == Int32(1)) && (LP_row == height)
            tableau[row + slack_count - height + Int32(1), 1] = -1.0
        end

        thread += stride
    end
    sync_threads()
    return nothing
end
=#