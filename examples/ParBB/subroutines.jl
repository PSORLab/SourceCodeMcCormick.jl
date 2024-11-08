
import EAGO: optimize_hook!
# Now on to the overloading of the B&B algorithm.
function EAGO.optimize_hook!(t::ExtendGPU, m::Optimizer)    
    m._global_optimizer._parameters.branch_variable = m._parameters.branch_variable
    EAGO.initial_parse!(m)

    optimize_gpu!(m)
end

function optimize_gpu!(m::Optimizer{R,S,Q}) where {R,S,Q<:EAGO.ExtensionType}
    solve_gpu!(m._global_optimizer)
    EAGO.unpack_global_solution!(m)
    return
end

"""
    global_solve_gpu_multi!

This version gets information about multiple nodes before running the lower problem solves
in parallel on the GPU. Information is successively added to a buffer in the DynamicExtGPU
extension which eventually solves the lower problems in parallel.

Upper problems are still solved using an ODERelaxProb and following the same techniques
as in the "normal" DynamicExt extension.
"""
function solve_gpu!(m::EAGO.GlobalOptimizer)
    # Turn off garbage collection
    GC.enable(false)

    # Identify the extension
    ext = EAGO._ext(m)

    # Set node count to 1
    m._node_count = 1

    # Prepare to run branch-and-bound
    EAGO.parse_global!(m)
    EAGO.presolve_global!(m)
    EAGO.print_preamble!(m)

    # Run the NLP solver to get a start-point upper bound with multi-starting (In development)
    multistart_upper!(m)

    # Fill the stack with multiple nodes for the GPU to parallelize
    prepopulate!(m)

    # Pre-allocate storage vectors
    ext.node_storage = Vector{EAGO.NodeBB}(undef, ext.node_limit)
    ext.lower_bound_storage = Vector{Float64}(undef, ext.node_limit)


    # Run branch and bound; terminate when the stack is empty or when some
    # tolerance or limit is hit
    while !EAGO.termination_check(m)

        # Update iteration counter
        m._iteration_count += 1
        
        # Garbage collect every gc_freq iterations
        if mod(m._iteration_count, EAGO._ext(m).gc_freq)==0
            GC.enable(true)
            GC.gc(false)
            GC.enable(false)
        end

        # Fathoming step
        EAGO.fathom!(m)

        # Extract up to `node_limit` nodes from the main problem stack
        count = min(ext.node_limit, m._node_count)
        ext.node_storage[1:count] .= EAGO.popmin!(m._stack, count)
        
        for i = 1:count
            ext.all_lvbs[i,:] .= ext.node_storage[i].lower_variable_bounds
            ext.all_uvbs[i,:] .= ext.node_storage[i].upper_variable_bounds
        end
        ext.node_len = count
        m._node_count -= count

        # Solve all the nodes in parallel
        lower_and_upper_problem!(m)
        
        EAGO.print_results!(m, true)

        for i in 1:ext.node_len
            make_current_node!(m)

            # Check for infeasibility and store the solution
            if m._lower_feasibility && !EAGO.convergence_check(m) 
                EAGO.print_results!(m, false)
                EAGO.store_candidate_solution!(m)

                # Perform post processing on each node I'm keeping track of
                # postprocess_total += @elapsed EAGO.postprocess!(m)
                # EAGO.postprocess!(m)
                # m._last_postprocessing_time += @elapsed EAGO.postprocess!(m)

                # Branch the nodes if they're feasible
                # if m._postprocess_feasibility
                EAGO.branch_node!(m)
                # end
            end
        end
        EAGO.set_global_lower_bound!(m)
        m._run_time = time() - m._start_time
        m._time_left = m._parameters.time_limit - m._run_time
        EAGO.log_iteration!(m)
        EAGO.print_iteration!(m, false)
    end
    EAGO.print_iteration!(m, true)

    EAGO.set_termination_status!(m)
    EAGO.set_result_status!(m)
    EAGO.print_solution!(m)

    # Turn back on garbage collection
    GC.enable(true)
    GC.gc()
end

# Helper functions here
function prepopulate!(t::ExtendGPU, m::EAGO.GlobalOptimizer)
    if t.prepopulate == true
        # println("Prepopulating with $(t.node_limit) total nodes")

        # Calculate the base number of splits we need for each parameter
        splits = floor(t.node_limit^(1/t.np))

        # Add splits to individual parameters as long as we don't go over
        # the limit
        split_groups = splits*ones(t.np)
        tracker = splits^(t.np)
        for i = 1:t.np
            if tracker * (split_groups[i]+1)/(split_groups[i]) < t.node_limit
                tracker = tracker * (split_groups[i]+1)/(split_groups[i])
                split_groups[i] += 1
            end
        end

        # Now we have the correct number of split groups, but we still have to
        # find the break points and add in additional nodes
        main_node = EAGO.popmin!(m._stack)
        lvbs = main_node.lower_variable_bounds
        uvbs = main_node.upper_variable_bounds
        split_vals = (uvbs .- lvbs)./(split_groups)

        # Now we create nodes, adding in additional points as necessary
        split_tracker = zeros(t.np)
        lbd = copy(lvbs)
        ubd = copy(uvbs)
        additional_points = t.node_limit - tracker
        for i = 1:tracker
            for j = 1:t.np
                lbd[j] = lvbs[j] + (split_tracker[j])*split_vals[j]
                ubd[j] = lvbs[j] + (split_tracker[j]+1)*split_vals[j]
            end
            if additional_points > 0
                # For additional points, just split the node in half at an arbitrary
                # parameter (and use modulo so it cycles through parameters)
                adjust = Int((i % t.np)+1)
                midL = copy(lbd)
                midU = copy(ubd)
                midL[adjust] = (lbd[adjust] + ubd[adjust])/2
                midU[adjust] = (lbd[adjust] + ubd[adjust])/2
                push!(m._stack, EAGO.NodeBB(copy(lbd), copy(midU), main_node.is_integer, main_node.continuous,
                                            max(main_node.lower_bound, m._lower_objective_value),
                                            min(main_node.upper_bound, m._upper_objective_value),
                                            1, main_node.cont_depth, i, EAGO.BD_NEG, 1, 0.0))
                push!(m._stack, EAGO.NodeBB(copy(midL), copy(ubd), main_node.is_integer, main_node.continuous,
                                            max(main_node.lower_bound, m._lower_objective_value),
                                            min(main_node.upper_bound, m._upper_objective_value),
                                            1, main_node.cont_depth, i, EAGO.BD_POS, 1, 0.0))
                additional_points -= 1
            else
                pos_or_neg = (i - (tracker/2)) < 0 ? EAGO.BD_NEG : EAGO.BD_POS
                push!(m._stack, EAGO.NodeBB(copy(lbd), copy(ubd), main_node.is_integer, main_node.continuous,
                                            max(main_node.lower_bound, m._lower_objective_value),
                                            min(main_node.upper_bound, m._upper_objective_value),
                                            1, main_node.cont_depth, i, pos_or_neg, 1, 0.0))
            end                    
            split_tracker[1] += 1
            for j = 1:(t.np-1)
                if split_tracker[j] == split_groups[j]
                    split_tracker[j] = 0
                    split_tracker[j+1] += 1
                end
            end
        end
        m._node_count = t.node_limit
    end
end
prepopulate!(m::EAGO.GlobalOptimizer{R,S,Q}) where {R,S,Q<:EAGO.ExtensionType} = prepopulate!(EAGO._ext(m), m)

"""
$(TYPEDSIGNATURES)

Adds the global optimizer's current node to the lower problem, and
extracts information so that the extension can solve the Ensemble
Problem.
"""
function add_to_substack!(t::ExtendGPU, m::EAGO.GlobalOptimizer)
    # For now, we'll assume that all points are feasible. This can be
    # changed in the future, once non-trivial constraints are considered

    # Add the current node to ExtendGPU's internal stack
    t.node_storage[t.node_len+1] = m._current_node

    # Add the current node's lower and upper variable bounds to the storage
    # to pass to the gpu
    t.all_lvbs[t.node_len+1, :] = m._current_node.lower_variable_bounds
    t.all_uvbs[t.node_len+1, :] = m._current_node.upper_variable_bounds
    
    # Increment the node length tracker
    t.node_len += 1
    return nothing
end
add_to_substack!(m::EAGO.GlobalOptimizer{R,S,Q}) where {R,S,Q<:EAGO.ExtensionType} = add_to_substack!(EAGO._ext(m), m)

# Solve the lower and upper problem for all nodes simultaneously, using the convex_func
# function from the ExtendGPU extension
function lower_and_upper_problem!(t::PointwiseGPU, m::EAGO.GlobalOptimizer)
    # Step 1) Bring the bounds into the GPU
    lvbs_d = CuArray(t.all_lvbs) 
    uvbs_d = CuArray(t.all_uvbs) # [points x num_vars]

    # Step 2) Preallocate points to evaluate 
    l, w = size(t.all_lvbs) #points, num_vars
    np = 2*w+2 #Adding an extra for upper bound calculations
    eval_points = Vector{CuArray{Float64}}(undef, 3*w)
    for i = 1:w
        eval_points[3i-2] = CuArray{Float64}(undef, l*np)
        eval_points[3i-1] = repeat(lvbs_d[:,i], inner=np)
        eval_points[3i] = repeat(uvbs_d[:,i], inner=np)
    end
    evals_d = CuArray{Float64}(undef, l*np)
    results_d = CuArray{Float64}(undef, l)
    
    # Step 3) Fill in each of these points
    for i = 1:w
        eval_points[3i-2][1:np:end] .= (lvbs_d[:,i].+uvbs_d[:,i])./2
        for j = 2:np-1 
            if j==2i
                eval_points[3i-2][j:np:end] .= (lvbs_d[:,i].+uvbs_d[:,i])./2 .+ t.α.*(uvbs_d[:,i].-lvbs_d[:,i])./2
            elseif j==2i+1
                eval_points[3i-2][j:np:end] .= (lvbs_d[:,i].+uvbs_d[:,i])./2 .- t.α.*(uvbs_d[:,i].-lvbs_d[:,i])./2
            else
                eval_points[3i-2][j:np:end] .= (lvbs_d[:,i].+uvbs_d[:,i])./2
            end
        end
        # Now we do np:np:end. Each one is set to the center of the variable bounds,
        # creating a degenerate interval. This gives us the upper bound.
        eval_points[3i-2][np:np:end] .= (lvbs_d[:,i].+uvbs_d[:,i])./2
        eval_points[3i-1][np:np:end] .= (lvbs_d[:,i].+uvbs_d[:,i])./2
        eval_points[3i][np:np:end] .= (lvbs_d[:,i].+uvbs_d[:,i])./2
    end

    # Step 4) Prepare the input vector for the convex function
    input = Vector{CuArray{Float64}}(undef, 0)
    for i = 1:w
        push!(input, [eval_points[3i-2], eval_points[3i-2], eval_points[3i-1], eval_points[3i]]...)
    end

    # Step 5) Perform the calculations
    evals_d .= t.convex_func(input...) #This should return a CuArray of the evaluations, same length as each input

    # Step 6) Extract the results and return to the CPU by filling in the lower/upper bound storages
    results_d .= evals_d[1:np:end]
    for i = 1:w
        results_d .-= (max.(evals_d[2i:np:end], evals_d[2i+1:np:end]).-evals_d[1:np:end])./t.α
    end

    t.lower_bound_storage .= Array(results_d)
    t.upper_bound_storage .= Array(evals_d[np:np:end])

    less_val = t.upper_bound_storage .< m._global_upper_bound
    if any(==(true), less_val)
        println("Should be lower")
        @show t.upper_bound_storage[less_val]
    end

    return nothing
end
lower_and_upper_problem!(m::EAGO.GlobalOptimizer{R,S,Q}) where {R,S,Q<:EAGO.ExtensionType} = lower_and_upper_problem!(EAGO._ext(m), m)


# A separate version of the lower_and_upper_problem! function that uses subgradients
function lower_and_upper_problem!(t::SubgradGPU, m::EAGO.GlobalOptimizer)
    # Step 1) Bring the bounds into the GPU
    lvbs_d = CuArray(t.all_lvbs)
    uvbs_d = CuArray(t.all_uvbs) # [points x num_vars]

    # Step 2) Preallocate points to evaluate 
    l, w = size(t.all_lvbs) #points, num_vars
    np = 2*w+2 #Adding an extra for upper bound calculations
    eval_points = Vector{CuArray{Float64}}(undef, 3*w) #Only 3x because one is repeated
    for i = 1:w
        eval_points[3i-2] = CuArray{Float64}(undef, l*np)
        eval_points[3i-1] = repeat(lvbs_d[:,i], inner=np)
        eval_points[3i] = repeat(uvbs_d[:,i], inner=np)
    end
    bounds_d = CuArray{Float64}(undef, l*np)
    
    # Step 3) Fill in each of these points
    for i = 1:w #1-3
        eval_points[3i-2][1:np:end] .= (lvbs_d[:,i].+uvbs_d[:,i])./2
        for j = 2:np-1 #2-7
            if j==2i
                eval_points[3i-2][j:np:end] .= (lvbs_d[:,i].+uvbs_d[:,i])./2 .+ t.α.*(uvbs_d[:,i].-lvbs_d[:,i])./2
            elseif j==2i+1
                eval_points[3i-2][j:np:end] .= (lvbs_d[:,i].+uvbs_d[:,i])./2 .- t.α.*(uvbs_d[:,i].-lvbs_d[:,i])./2
            else
                eval_points[3i-2][j:np:end] .= (lvbs_d[:,i].+uvbs_d[:,i])./2
            end
        end
        # Now we do np:np:end. Each one is set to the center of the variable bounds,
        # creating a degenerate interval. This gives us the upper bound.
        eval_points[3i-2][np:np:end] .= (lvbs_d[:,i].+uvbs_d[:,i])./2
        eval_points[3i-1][np:np:end] .= (lvbs_d[:,i].+uvbs_d[:,i])./2
        eval_points[3i][np:np:end] .= (lvbs_d[:,i].+uvbs_d[:,i])./2
    end

    # Step 4) Prepare the input vector for the convex function
    input = Vector{CuArray{Float64}}(undef, 0)
    for i = 1:w
        push!(input, [eval_points[3i-2], eval_points[3i-2], eval_points[3i-1], eval_points[3i]]...)
    end

    # Step 5) Perform the calculations
    func_output = t.convex_func_and_subgrad(input...) # n+2-dimensional

    # Step 6) Use values and subgradients to calculate lower bounds
    bounds_d .= func_output[1]
    for i = 1:w
        bounds_d .+= -(func_output[i+2] .>= 0.0).*func_output[i+2].*(eval_points[3i-2][:] .- eval_points[3i-1][:]) .- 
                      (func_output[i+2] .<= 0.0).*func_output[i+2].*(eval_points[3i-2][:] .- eval_points[3i][:])
    end

    # Add to lower and upper bound storage
    t.lower_bound_storage .= max.(Array(func_output[2][1:np:end]), [maximum(bounds_d[i:i+np-2]) for i in 1:np:l*np])
    t.upper_bound_storage .= Array(bounds_d[np:np:end])

    return nothing
end

# A third version of lower_and_upper_problem! that uses the new GPU Simplex algorithm
function lower_and_upper_problem_old!(t::SimplexGPU_OnlyObj, m::EAGO.GlobalOptimizer)
    # Step 1) Bring the bounds into the GPU
    lvbs_d = CuArray(t.all_lvbs)
    uvbs_d = CuArray(t.all_uvbs) # [points x num_vars]

    # Step 2) Preallocate points to evaluate 
    l, w = size(t.all_lvbs) #points, num_vars
    np = 2*w+2 #Number of points; Adding an extra for upper bound calculations
    eval_points = Vector{CuArray{Float64}}(undef, 3*w) #Only 3x because one is repeated
    for i = 1:w
        eval_points[3i-2] = CuArray{Float64}(undef, l*np)
        eval_points[3i-1] = repeat(lvbs_d[:,i], inner=np)
        eval_points[3i] = repeat(uvbs_d[:,i], inner=np)
    end
    bounds_d = CuArray{Float64}(undef, l)
    
    # Step 3) Fill in each of these points
    for i = 1:w #1-3
        eval_points[3i-2][1:np:end] .= (lvbs_d[:,i].+uvbs_d[:,i])./2
        for j = 2:np-1 #2-7
            if j==2i
                eval_points[3i-2][j:np:end] .= (lvbs_d[:,i].+uvbs_d[:,i])./2 .- t.α.*(uvbs_d[:,i].-lvbs_d[:,i])./2
            elseif j==2i+1
                eval_points[3i-2][j:np:end] .= (lvbs_d[:,i].+uvbs_d[:,i])./2 .+ t.α.*(uvbs_d[:,i].-lvbs_d[:,i])./2
            else
                eval_points[3i-2][j:np:end] .= (lvbs_d[:,i].+uvbs_d[:,i])./2
            end
        end
        # Now we do np:np:end. Each one is set to the center of the variable bounds,
        # creating a degenerate interval. This gives us the upper bound.
        eval_points[3i-2][np:np:end] .= (lvbs_d[:,i].+uvbs_d[:,i])./2
        eval_points[3i-1][np:np:end] .= (lvbs_d[:,i].+uvbs_d[:,i])./2
        eval_points[3i][np:np:end] .= (lvbs_d[:,i].+uvbs_d[:,i])./2
    end

    # Step 4) Prepare the input vector for the convex function
    input = Vector{CuArray{Float64}}(undef, 0)
    for i = 1:w
        push!(input, [eval_points[3i-2], eval_points[3i-2], eval_points[3i-1], eval_points[3i]]...)
    end

    # Step 5) Perform the calculations
    func_output = t.convex_func_and_subgrad(input...) # n+2-dimensional
    # Also need whatever constraints!!

    # Step 6) Use values and subgradients to prepare the stacked Simplex tableau

    # First things first, we can prepare the "b" vector and see if we need any auxiliary systems.
    # This step calculates the intercept of b at x=x_lo, which is equivalent to calculating
    # the intercept at x=0 and then later shifting x_lo to 0, but without the extra re-calculation
    # steps
    b_start = func_output[1]
    for i = 1:w
        b_start -= func_output[i+2].*(eval_points[3i-2] .- eval_points[3i-1])

        # func_output[i+2] is the subgradient of the convex relaxation in the i'th dimension
        # eval_points[3i-2] is the cv/cc point used to obtain the relaxation
        # eval_points[3i-1] is the lower bound for this relaxation
        # eval_points[3i] is the upper bound (which isn't used here)

        # Note that <= [upper bound] will change to <= [upper bound] - [lower bound] 
        # for each variable, later
    end

    if all(<=(0.0), b_start) 
        #If b_start is all nonpositive, we don't need any auxiliary systems

        # Start making the tableau as normal, since we have a basic feasible solution at the start.
        # Create an extended b_array 
        b_array = vcat([vcat(-b_start[np*(j-1)+1:np*j-1],   # First "1:(np-1)" points for each node
                             uvbs_d[j,:].-lvbs_d[j,:],      # Upper bound minus lower bound 
                             0.0)                           # 0.0 for the objective function row
                        for j=1:l]...) # Repeat for every node
        
        # Prepare the epigraph variable columns. We're minimizing "Z", but since "Z" is unbounded, we
        # convert it to Z = Z_pos - Z_neg, where Z_pos, Z_neg >= 0.0. The first column will be Z_pos,
        # and the second column will be Z_neg. The upper bound rows and auxiliary system objective
        # function row will have these as 0; the objective function row will be [1, -1] (minimizing
        # Z_pos - Z_neg); and the constraints associated with Z will be [-1, 1] (-Z = -Z_pos + Z_neg)
        epigraph = hcat(-CUDA.ones(Float64, length(b_array)), CUDA.ones(Float64, length(b_array)))
        for i = 1:w
            # Starting at the first upper bound, repeat for every tableau
            epigraph[np+i-1 : np+w : end, :] .= 0.0
        end
        epigraph[np+w : np+w : end, :] .*= -1.0 # The main objective function is opposite the other rows (minimizing Z)
        epigraph[np+w+1 : np+w : end, :] .= 0.0 # The epigraph column is 0 in the auxiliary objective row


        # Combine the epigraph columns, "A" matrix, slack variable columns, and "b" array into the stacked tableaus
        tableaus = hcat(epigraph,                                                       # Epigraph variable columns
                        [vcat([vcat(func_output[i+2][np*(j-1)+1:np*j-1],                    # >>Subgradient values for the i'th variable for the j'th node
                            CUDA.zeros(Float64, w),                                         # >>Zeros for upper bound constraints (will fill in later with 1.0s)
                            0.0)                                                            # >>0.0 for the objective function row
                            for j = 1:l]...)                                                # Repeat for every j'th node vertically
                        for i = 1:w]...,                                                # Add a column for every i'th variable
                        [CUDA.zeros(Float64, length(b_array)) for _ = 1:(np-1)+w]...,   # Slack variables (will fill in later with 1.0s)
                        b_array)                                                        # The array of b's
        
        # Fill in the upper bound constraint indices and the slack variables
        for i = 1:w
            tableaus[np+i-1 : np+w : end, i+2] .= 1.0
        end
        for i = 1:(np-1)+w
            tableaus[i:np+w:end, (w+2)+i] .= 1.0
        end

        tableaus .= parallel_simplex(tableaus, np+w)

    else
        # It was detected that some slack variable coefficient would be negative, so we need to make auxiliary systems. 
        # Note: It's probably worth it to only do auxiliary systems for the tableaus that will need it. At least check 
        # to see how common this is, and whether it'll be necessary...

        # Create the extended b_array
        b_array = vcat([vcat(-b_start[np*(j-1)+1:np*j-1],   # First "1:(np-1)" points for each node
                             uvbs_d[j,:].-lvbs_d[j,:],      # Upper bound minus lower bound 
                             0.0,                           # 0.0 for the objective function row
                             0.0)                           # 0.0 for the auxiliary system objective function row
                        for j=1:l]...) # Repeat for every node

        # (NOTE: Should we be scaling all the variables/subgradients so that the variables are bounded on [0, 1]?)

        # Prepare the epigraph variable columns. We're minimizing "Z", but since "Z" is unbounded, we
        # convert it to Z = Z_pos - Z_neg, where Z_pos, Z_neg >= 0.0. The first column will be Z_pos,
        # and the second column will be Z_neg. The upper bound rows and auxiliary system objective
        # function row will have these as 0; the objective function row will be [1, -1] (minimizing
        # Z_pos - Z_neg); and the constraints associated with Z will be [-1, 1] (-Z = -Z_pos + Z_neg)
        epigraph = hcat(-CUDA.ones(Float64, length(b_array)), CUDA.ones(Float64, length(b_array)))
        for i = 1:w
            # Starting at the first upper bound, repeat for every tableau 
            # (which has np+w+1 rows thanks to the auxiliary row)
            epigraph[np+i-1 : np+w+1 : end, :] .= 0.0 
        end
        epigraph[np+w : np+w+1 : end, :] .*= -1.0 # The main objective function is opposite the other rows (minimizing Z)
        epigraph[np+w+1 : np+w+1 : end, :] .= 0.0 # The epigraph column is 0 in the auxiliary objective row

        # Combine the epigraph columns, "A" matrix, slack variable columns, and "b" array into the stacked tableaus
        tableaus = hcat(epigraph,                                                         # Epigraph variable columns
                        [vcat([vcat(func_output[i+2][np*(j-1)+1:np*j-1],                    # >>Subgradient values for the i'th variable for the j'th node
                            CUDA.zeros(Float64, w),                                         # >>Zeros for upper bound constraints (will fill in later with 1.0s)
                            0.0,                                                            # >>0.0 for the objective function row
                            0.0)                                                            # >>0.0 for the auxiliary objective function row
                            for j = 1:l]...)                                                # Repeat for every j'th node vertically
                        for i = 1:w]...,                                                  # Add a column for every i'th variable
                        [CUDA.zeros(Float64, length(b_array)) for _ = 1:2*((np-1)+w)]..., # Slack and auxiliary variables (will fill in later with 1.0s)
                        b_array)                                                          # The array of b's
        
        # Fill in the upper bound constraint indices
        for i = 1:w
            tableaus[np+i-1 : np+w+1 : end, i+2] .= 1.0 #np+w+1 length now, because of the auxiliary row
        end

        # Fill in the slack variables like normal, and then add auxiliary variables as needed
        signs = sign.(tableaus[:,end])
        signs[signs.==0] .= 1.0
        for i = 1:np+w-1
            tableaus[i:np+w+1:end, (w+2)+i] .= 1.0 #np+w+1 length now, because of the auxiliary row

            # If the "b" row is negative, do the following:
            # 1) Flip the row so that "b" is positive
            # 2) Subtract the entire row FROM the auxiliary objective row
            # 3) Add an auxiliary variable for this row
            tableaus[i:np+w+1:end, :] .*= signs[i:np+w+1:end] #Flipped the row if b was negative
            tableaus[np+w+1 : np+w+1 : end, :] .-= (signs[i:np+w+1:end].<0.0).*tableaus[i:np+w+1:end, :] #Row subtracted from auxiliary objective row
            tableaus[i:np+w+1:end, (w+2)+np+w-1+i] .+= (signs[i:np+w+1:end].<0.0).*1.0
        end

        # Send the tableaus to the parallel_simplex algorithm, with the "aux" flag set to "true"
        tableaus .= parallel_simplex(tableaus, np+w+1, aux=true)

        if all(abs.(tableaus[np+w+1:np+w+1:end,end]).<=1E-10)
            # Delete the [np+w+1 : np+w+1 : end] rows and the [w+1+(np+w-1) + 1 : end-1] columns
            # Note: is it faster to NOT remove the rows/columns and just have an adjusted simplex
            # algorithm that ignores them? Maybe, maybe not. I'll test later.
            tableaus = tableaus[setdiff(1:end, np+w+1:np+w+1:end), setdiff(1:end, w+2+(np+w-1):end-1)]
            tableaus .= parallel_simplex(tableaus, np+w)
        else
            warn = true
        end
    end

    # display(Array(func_output[2]))
    # display(Array(tableaus))
    # display(Array(-tableaus[np+w:np+w:end,end]))
    # display(Array(func_output[2][1:np:end]))
    # display(Array(max.(func_output[2][1:np:end], -tableaus[np+w:np+w:end,end])))

    # Step 8) Add results to lower and upper bound storage
    t.lower_bound_storage .= Array(max.(func_output[2][1:np:end], -tableaus[np+w:np+w:end,end]))
    t.upper_bound_storage .= Array(func_output[1][np:np:end])

    return nothing
end

# An even newer Simplex
function lower_and_upper_problem_slightly_old!(t::SimplexGPU_OnlyObj, m::EAGO.GlobalOptimizer)
    # Step 1) Bring the bounds into the GPU
    lvbs_d = CuArray(t.all_lvbs)
    uvbs_d = CuArray(t.all_uvbs) # [points x num_vars]

    # Step 2) Set up points to evaluate, which are the centers of every node
    l, w = size(t.all_lvbs) #points, num_vars
    np = 1 #Number of evaluations per node; Adding an extra for upper bound calculations
    eval_points = Vector{CuArray{Float64}}(undef, 3*w) #Only 3x because cv is the same as cc
    temp_lvbs = CuArray{Float64}(undef, l) #Pre-allocate slices of lvbs
    temp_uvbs = CuArray{Float64}(undef, l) #Pre-allocate slices of uvbs
    for i = 1:w
        # Temporarily hold slices of variable bounds
        temp_lvbs .= lvbs_d[:,i]
        temp_uvbs .= uvbs_d[:,i]

        # Set up bounds to evaluate
        eval_points[3i-1] = repeat(temp_lvbs, inner=np)
        eval_points[3i] = repeat(temp_uvbs, inner=np)

        # Calculate midpoints of the bounds
        eval_points[3i-2] = (eval_points[3i-1].+eval_points[3i])./2

        # Correct the bounds for the upper bound calculation (every 2 evaluations)
        # eval_points[3i-1][np:np:end] .= eval_points[3i-2][np:np:end]
        # eval_points[3i][np:np:end] .= eval_points[3i-2][np:np:end]
    end
    # println("After initial setup:")
    # CUDA.memory_status();println("")

    # Step 3) Perform the calculations (Note: also need to add in constraint handling. Perhaps
    #         that will be a different function, so that this one can stay as-is?
    func_output = t.convex_func_and_subgrad(
                ([[eval_points[3i-2],eval_points[3i-2],eval_points[3i-1],eval_points[3i]] for i=1:w]...)...) # n+2-dimensional

    # println("After calling the function:")
    # CUDA.memory_status();println("")
    # Might as well save the upper bound right now, since we have it
    t.upper_bound_storage .= Array(func_output[1][np:np:end])
    # println("Saving upper bound results:")
    # CUDA.memory_status();println("")

    # Step 4) Use values and subgradients to prepare the stacked Simplex tableau.
    # Based on this procedure, we should never need an auxiliary system? Check on that to
    # be sure, because if we don't need an auxiliary system, that's much easier

    # Preallocate subgradient matrices and the b vector
    subgradients = CuArray{Float64}(undef, l, w)
    corrected_subgradients = CuArray{Float64}(undef, l, w)
    b_val = CuArray{Float64}(undef, l, 1)
    # println("More preallocations:")
    # CUDA.memory_status();println("")

    # Extract subgradients and apply a correction for shifting variables to [0,1]
    subgradients .= hcat([func_output[i+2][1:np:end] for i in 1:w]...) #Only for the lower bound, not upper bound
    corrected_subgradients .= (subgradients).*(uvbs_d .- lvbs_d)

    # Calculate corrected "b" values based on the intercept at the evaluation point
    # and the corrections for shifted variables
    b_val .= sum(hcat([eval_points[3i-2][1:np:end] for i=1:w]...).*subgradients, dims=2) .- func_output[1][1:np:end] .- sum(lvbs_d.*subgradients, dims=2)

    # If there are any negative values in b, we can simply change the epigraph
    # variable to be that much higher to make the minimum 0. Note that this
    # won't work for constraints, it only works because it's for the objective
    # function and we have "z" in the tableau.
    add_val = 0.0
    if any(<(0.0), b_val)
        add_val = -minimum(b_val)
        b_val .+= add_val
    end

    # Free up eval_points since we no longer need it
    CUDA.unsafe_free!.(eval_points)

    # Preemptively determine how many slack variables we'll need. This is going to be
    # the total number of cuts (an input to this function), plus one for the lower bound
    # value, plus the number of variables (since each has an upper bound of 1)
    slack_count = t.max_cuts+1+w # n_cuts, lower bound, w [upper bounds]
    
    # Create the stacked tableau as a big array of 0's, and then we fill it in as necessary.
    tableau = CUDA.zeros(Float64, l*(slack_count+1), 1+w+slack_count+1)
    solution_tableau = similar(tableau)
    # println("Tableau prepared:")
    # CUDA.memory_status();println("")

    # Fill in the first column, corresponding to the epigraph variable
    tableau[1:(slack_count+1):end,1] .= 1.0 # Lower bound constraint
    tableau[2:(slack_count+1):end,1] .= 1.0 # Only one cut to consider for now
    tableau[(slack_count+1):(slack_count+1):end,1] .= -1.0 # Objective row

    # Fill in the corrected subgradients in their respective columns
    tableau[2:(slack_count+1):end,2:1+w] .= corrected_subgradients

    # Fill in the slack variables (besides the rows for cuts we haven't done yet)
    for i = 1:slack_count
        if (i<=t.max_cuts+1) && (t.max_cuts>1) && (i>2)  #Reserving rows 3 to max_cuts+1 for future cuts
            continue
        end
        tableau[i:(slack_count+1):end,1+w+i] .= 1.0
    end

    # Fill in the lower bound for the epigraph variable
    tableau[1:(slack_count+1):end,end] .= -func_output[2][1:np:end] .+ add_val

    # Fill in the value for the first cut (b)
    tableau[2:(slack_count+1):end,end] .= b_val

    # Fill in the upper bounds for all the variables, which are all 1's
    for i = 1:w
        tableau[t.max_cuts+1+i:(slack_count+1):end,1+i] .= 1.0 # The variable itself
        tableau[t.max_cuts+1+i:(slack_count+1):end,end] .= 1.0 # The variable's upper bound (always 1 because we shifted it)
    end

    # Make sure all the rightmost column values are positive
    if any(<(0.0), tableau[:,end])
        # display(tableau[:,end])
        error("Check b_val, might need an auxiliary system (or more creativity)")
    end
    # println("Tableau filled:")
    # CUDA.memory_status();println("")

    # Free up the func outputs since we no longer need them
    CUDA.unsafe_free!.(func_output)

    # Pass the tableau through the simplex algorithm and see what we get out of it
    # (Note that the solution values will be negated. That's fine, we don't actually
    # care what they are just yet.)
    solution_tableau .= tableau
    # println("Right before simplex:")
    # CUDA.memory_status();println("")
    parallel_simplex(solution_tableau, slack_count+1)

    # println("Right after simplex:")
    # CUDA.memory_status();println("")

    # Now we need to add another cut, which means we have to extract out the solution 
    # from the tableau, convert back into un-shifted variables, pass it back through
    # the convex evaluator, and add rows to the tableau.

    # Preallocate some arrays checks we'll be using 
    tableau_vals = CuArray{Float64}(undef, l,(slack_count+1))
    variable_vals = CuArray{Float64}(undef, l,(slack_count+1))
    bool_check = CuArray{Bool}(undef, l,(slack_count+1))
    zero_check = CuArray{Bool}(undef, l,(slack_count+1))
    short_eval_points = Vector{CuArray{Float64}}(undef, w) # Only need pointwise evaluations
    # println("Preallocations for next cuts:")
    # CUDA.memory_status();println("")

    for cut = 1:t.max_cuts-1 # Two additional cuts.
        # Extract solution values from the tableau to decide where the evaluations are.
        # How to do this... Search through [2:w+1] columns to find columns that are all 0's with
        # single 1's. 
        # Maybe think about this for an individual block. We have a block of size (slack_count+1, w),
        # and we need to identify which rows are relevant. We could go by row, but then we'd have
        # to get to the end before we could identify anything... we could preallocate space for
        # the variable values, at least---or, wait, that's already done with eval_points[3i-2].
        # So that's nice. Uhhh... okay... 
        # Figure out which ones are "correct" for each variable?
        for i = 1:w
            temp_lvbs .= lvbs_d[:,i]
            temp_uvbs .= uvbs_d[:,i]
            tableau_vals .= reshape(solution_tableau[:,end], l,(slack_count+1))
            variable_vals .= reshape(solution_tableau[:,1+i], l,(slack_count+1))
            bool_check .= (variable_vals .== 1.0)
            zero_check .= (variable_vals .== 0.0)
            bool_check .&= (count(bool_check, dims=2).==1)
            bool_check .&= (count(zero_check, dims=2).==slack_count)
            tableau_vals .*= bool_check
            short_eval_points[i] = min.(0.95, max.(0.05, sum(tableau_vals, dims=2))).*(temp_uvbs .- temp_lvbs).+temp_lvbs
        end
        # println("Cut $cut, found short eval points:")
        # CUDA.memory_status();println("")

        # Okay, so now we have the points to evaluate, we need to call the function again
        func_output = t.convex_func_and_subgrad(
            ([[short_eval_points[i],short_eval_points[i],lvbs_d[:,i],uvbs_d[:,i]] for i=1:w]...)...) # n+2-dimensional

        # println("Function called again:")
        # CUDA.memory_status();println("")
        # As before, calculate corrected subgradients and the new b_values
        subgradients .= hcat([func_output[i+2] for i in 1:w]...) #Only for the lower bound, not upper bound
        corrected_subgradients .= (subgradients).*(uvbs_d .- lvbs_d)
        b_val .= sum(hcat([short_eval_points[i] for i=1:w]...).*subgradients, dims=2) .- func_output[1] .- sum(lvbs_d.*subgradients, dims=2)

        # Add in the extra factor
        b_val .+= add_val

        # If any of b_val is [still] negative, update add_val and the rest of the tableau
        if any(<(0.0), b_val)
            update = -minimum(b_val)
            b_val .+= update
            tableau[1:slack_count+1:end, end] .+= update
            for i = 2:(2+cut-1)
                tableau[i:slack_count+1:end, end] .+= update
            end
            add_val += update
        end

        # Clear the short_eval_points from memory
        CUDA.unsafe_free!.(short_eval_points)

        # We can now place in these values into the tableau in the spots we left open earlier
        tableau[2+cut:slack_count+1:end, 1] .= 1.0
        tableau[2+cut:slack_count+1:end, 2:1+w] .= corrected_subgradients
        tableau[2+cut:slack_count+1:end, 1+w+2+cut] .= 1.0
        tableau[2+cut:slack_count+1:end, end] .= b_val

        # Adjust the final line of each problem to be the original minimization problem
        tableau[(slack_count+1):(slack_count+1):end,:] .= hcat(-CUDA.one(Float64), CUDA.zeros(Float64, 1, w+slack_count+1))

        # Run the simplex algorithm again
        solution_tableau .= tableau
        
        # println("Everything until simplex again")
        # CUDA.memory_status();println("")
        parallel_simplex(solution_tableau, slack_count+1)
        # println("Right after simplex again:")
        # CUDA.memory_status();println("")
    end
    
    # println("End of simplexing:")
    # CUDA.memory_status();println("")
    
    # Save the lower bounds
    t.lower_bound_storage .= Array(-(solution_tableau[slack_count+1:slack_count+1:end,end] .- add_val))

    # println("After lower bounds saved:")
    # CUDA.memory_status();println("")

    # Free variables we're finally done with
    for i in [lvbs_d, uvbs_d, temp_lvbs, temp_uvbs, subgradients, corrected_subgradients, b_val,
              tableau, solution_tableau, tableau_vals, variable_vals, bool_check, zero_check]
        CUDA.unsafe_free!(i)
    end
    # println("Freed up storage, and done.:")
    # CUDA.memory_status();println("")
    # error()
    return nothing
end

function lower_and_upper_problem_split!(t::SimplexGPU_OnlyObj, m::EAGO.GlobalOptimizer)
    # Step 1) Bring the bounds into the GPU
    lvbs_d = CuArray(t.all_lvbs)
    uvbs_d = CuArray(t.all_uvbs) # [points x num_vars]

    # Step 2) Set up points to evaluate, which are the centers of every node
    l, w = size(t.all_lvbs) #points, num_vars
    eval_points = (lvbs_d .+ uvbs_d)./2

    # Step 3) Perform the calculations (Note: also need to add in constraint handling. Perhaps
    #         that will be a different function, so that this one can stay as-is?

    # Upper bound calculations first; lower and upper bounds, and cv/cc, are the midpoints of nodes
    # NOTE: Speed can be improved by 75% for this call if you switch it out with a separate
    #       function that only calculates the lower bound, for example. Or maybe more if you can
    #       make a normal GPU-compatible version of the objective function.
    func_output = @views t.convex_func_and_subgrad(
                ([[eval_points[:,i],eval_points[:,i],eval_points[:,i],eval_points[:,i]] for i=1:w]...)...) # n+2-dimensional

    t.upper_bound_storage .= Array(func_output)

    # Free up the func outputs
    CUDA.unsafe_free!(func_output)

    # Now lower bound calculations. It's the same as upper bounds, but we use lvbs and uvbs.
    func_output = @views t.convex_func_and_subgrad(
        ([[eval_points[:,i],eval_points[:,i],lvbs_d[:,i],uvbs_d[:,i]] for i=1:w]...)...) # n+2-dimensional

    # Step 4) Use values and subgradients to prepare the stacked Simplex tableau.
    # Based on this procedure, we should never need an auxiliary system? Check on that to
    # be sure, because if we don't need an auxiliary system, that's much easier

    # Preallocate subgradient matrices and the b vector
    subgradients = CuArray{Float64}(undef, l, w)
    corrected_subgradients = CuArray{Float64}(undef, l, w)
    mid_times_sub = CuArray{Float64}(undef, l, w)
    low_times_sub = CuArray{Float64}(undef, l, w)
    b_val = CuArray{Float64}(undef, l, 1)

    # Extract subgradients and apply a correction for shifting variables to [0,1]
    subgradients .= hcat([func_output[i+2] for i in 1:w]...) 
    corrected_subgradients .= (subgradients).*(uvbs_d .- lvbs_d)

    # Calculate corrected "b" values based on the intercept at the evaluation point
    # and the corrections for shifted variables
    mid_times_sub .= eval_points.*subgradients
    low_times_sub .= lvbs_d.*subgradients
    b_val .= sum(mid_times_sub, dims=2) .- func_output[1] .- sum(low_times_sub, dims=2)

    # If there are any negative values in b, we can simply change the epigraph
    # variable to be that much higher to make the minimum 0. Note that this
    # won't work for constraints, it only works because it's for the objective
    # function and we have "z" in the tableau.
    add_val = 0.0
    if any(<(0.0), b_val)
        add_val = -minimum(b_val)
        b_val .+= add_val
    end

    # Preemptively determine how many slack variables we'll need. This is going to be
    # the total number of cuts (an input to this function), plus one for the lower bound
    # value, plus the number of variables (since each has an upper bound of 1)
    slack_count = t.max_cuts+1+w # n_cuts, lower bound, w [upper bounds]
    
    # Create the stacked tableau as a big array of 0's, and then we fill it in as necessary.
    tableau = CUDA.zeros(Float64, l*(slack_count+1), 1+w+slack_count+1)
    solution_tableau = similar(tableau)

    # Fill in the first column, corresponding to the epigraph variable
    tableau[1:(slack_count+1):end,1] .= 1.0 # Lower bound constraint
    tableau[2:(slack_count+1):end,1] .= 1.0 # Only one cut to consider for now
    tableau[(slack_count+1):(slack_count+1):end,1] .= -1.0 # Objective row

    # Fill in the corrected subgradients in their respective columns
    tableau[2:(slack_count+1):end,2:1+w] .= corrected_subgradients

    # Fill in the slack variables (besides the rows for cuts we haven't done yet)
    for i = 1:slack_count
        if (i<=t.max_cuts+1) && (t.max_cuts>1) && (i>2)  #Reserving rows 3 to max_cuts+1 for future cuts
            continue
        end
        tableau[i:(slack_count+1):end,1+w+i] .= 1.0
    end

    # Fill in the lower bound for the epigraph variable
    tableau[1:(slack_count+1):end,end] .= -func_output[2] .+ add_val

    # Fill in the value for the first cut (b)
    tableau[2:(slack_count+1):end,end] .= b_val

    # Fill in the upper bounds for all the variables, which are all 1's
    for i = 1:w
        tableau[t.max_cuts+1+i:(slack_count+1):end,1+i] .= 1.0 # The variable itself
        tableau[t.max_cuts+1+i:(slack_count+1):end,end] .= 1.0 # The variable's upper bound (always 1 because we shifted it)
    end

    # Make sure all the rightmost column values are positive (not necessary, and for some
    # reason this eats up GPU memory?)
    # if any((@view tableau[:,end]) .< 0.0)
    #     error("Check b_val, might need an auxiliary system (or more creativity)")
    # end

    # Free up the func outputs since we no longer need them
    CUDA.unsafe_free!.(func_output)
    

    # Pass the tableau through the simplex algorithm and see what we get out of it
    # (Note that the solution values will be negated. That's fine, we don't actually
    # care what they are just yet.)
    solution_tableau .= tableau
    parallel_simplex(solution_tableau, slack_count+1)

    # Now we need to add another cut, which means we have to extract out the solution 
    # from the tableau, convert back into un-shifted variables, pass it back through
    # the convex evaluator, and add rows to the tableau.

    # Preallocate some arrays checks we'll be using 
    tableau_vals = CuArray{Float64}(undef, l,(slack_count+1))
    variable_vals = CuArray{Float64}(undef, l,(slack_count+1))
    bool_check = CuArray{Bool}(undef, l,(slack_count+1))
    zero_check = CuArray{Bool}(undef, l,(slack_count+1))
    for cut = 1:t.max_cuts-1 # Two additional cuts.
        # Extract solution values from the tableau to decide where the evaluations are.
        # Figure out which ones are "correct" for each variable?
        for i = 1:w
            tableau_vals .= reshape((@view solution_tableau[:,end]), l,(slack_count+1))
            variable_vals .= reshape((@view solution_tableau[:,1+i]), l,(slack_count+1))
            bool_check .= (variable_vals .== 1.0)
            zero_check .= (variable_vals .== 0.0)
            bool_check .&= (count(bool_check, dims=2).==1)
            bool_check .&= (count(zero_check, dims=2).==slack_count)
            tableau_vals .*= bool_check
            eval_points[:,i] .= @views min.(0.95, max.(0.05, sum(tableau_vals, dims=2))).*(uvbs_d[:,i] .- lvbs_d[:,i]).+lvbs_d[:,i]
        end

        # Okay, so now we have the points to evaluate, we need to call the function again
        func_output = @views t.convex_func_and_subgrad(
            ([[eval_points[:,i],eval_points[:,i],lvbs_d[:,i],uvbs_d[:,i]] for i=1:w]...)...) # n+2-dimensional
        
        # As before, calculate corrected subgradients and the new b_values
        subgradients .= hcat([func_output[i+2] for i in 1:w]...) #Only for the lower bound, not upper bound
        corrected_subgradients .= (subgradients).*(uvbs_d .- lvbs_d)
        mid_times_sub .= eval_points.*subgradients
        low_times_sub .= lvbs_d.*subgradients
        b_val .= sum(mid_times_sub, dims=2) .- func_output[1] .- sum(low_times_sub, dims=2)

        # Add in the extra factor
        b_val .+= add_val

        # If any of b_val is [still] negative, update add_val and the rest of the tableau
        if any(<(0.0), b_val)
            update = -minimum(b_val)
            b_val .+= update
            tableau[1:slack_count+1:end, end] .+= update
            for i = 2:(2+cut-1)
                tableau[i:slack_count+1:end, end] .+= update
            end
            add_val += update
        end

        # Free up the func outputs since we no longer need them
        CUDA.unsafe_free!.(func_output)

        # We can now place in these values into the tableau in the spots we left open earlier
        tableau[2+cut:slack_count+1:end, 1] .= 1.0
        tableau[2+cut:slack_count+1:end, 2:1+w] .= corrected_subgradients
        tableau[2+cut:slack_count+1:end, 1+w+2+cut] .= 1.0
        tableau[2+cut:slack_count+1:end, end] .= b_val

        # Adjust the final line of each problem to be the original minimization problem
        tableau[(slack_count+1):(slack_count+1):end,:] .= hcat(-CUDA.one(Float64), CUDA.zeros(Float64, 1, w+slack_count+1))

        # Run the simplex algorithm again
        solution_tableau .= tableau
        parallel_simplex(solution_tableau, slack_count+1)
    end

    # Save the lower bounds (remembering to negate the values)
    t.lower_bound_storage .= @views Array(-(solution_tableau[slack_count+1:slack_count+1:end,end] .- add_val))

    for i in [lvbs_d, uvbs_d, eval_points, subgradients, corrected_subgradients, b_val,
              tableau, solution_tableau, tableau_vals, variable_vals, bool_check, zero_check]
        CUDA.unsafe_free!(i)
    end
    return nothing
end

function lower_and_upper_problem!(t::SimplexGPU_OnlyObj, m::EAGO.GlobalOptimizer)
    t.lower_counter += 1
    # Step 1) Bring the bounds into the GPU
    lvbs_d = CuArray(t.all_lvbs[1:t.node_len,:])
    uvbs_d = CuArray(t.all_uvbs[1:t.node_len,:]) # [points x num_vars]
    
    # Step 2) Set up points to evaluate, which are the centers of every node
    # l, w = size(t.all_lvbs) #points, num_vars
    w = t.np
    l = t.node_len
    eval_points = (lvbs_d .+ uvbs_d)./2

    # Step 3) Perform the calculations (Note: also need to add in constraint handling. Perhaps
    #         that will be a different function, so that this one can stay as-is?

    # Perform both lower and upper bound calculations, stacked on top of one another. This
    # is faster than splitting lower and upper bounding problems and calling the convex
    # function twice, 
    t.relax_time += @elapsed CUDA.@sync func_output = @views t.convex_func_and_subgrad(
        ([[[eval_points[:,i];eval_points[:,i]],[eval_points[:,i];eval_points[:,i]],[lvbs_d[:,i];eval_points[:,i]],[uvbs_d[:,i];eval_points[:,i]]] for i=1:w]...)...) # n+2-dimensional
    t.upper_bound_storage[1:t.node_len] .= @views Array(func_output[1][l+1:end])


    # Step 4) Use values and subgradients to prepare the stacked Simplex tableau.
    # Based on this procedure, we should never need an auxiliary system? Check on that to
    # be sure, because if we don't need an auxiliary system, that's much easier

    # Preallocate subgradient matrices and the b vector
    subgradients = CuArray{Float64}(undef, l, w)
    corrected_subgradients = CuArray{Float64}(undef, l, w)
    mid_times_sub = CuArray{Float64}(undef, l, w)
    low_times_sub = CuArray{Float64}(undef, l, w)
    b_val = CuArray{Float64}(undef, l, 1)

    # Extract subgradients and apply a correction for shifting variables to [0,1]
    subgradients .= @views hcat([func_output[i+2][1:l] for i in 1:w]...) 
    corrected_subgradients .= (subgradients).*(uvbs_d .- lvbs_d)

    # Calculate corrected "b" values based on the intercept at the evaluation point
    # and the corrections for shifted variables
    mid_times_sub .= eval_points.*subgradients
    low_times_sub .= lvbs_d.*subgradients
    b_val .= sum(mid_times_sub, dims=2) .- (@view func_output[1][1:l]) .- sum(low_times_sub, dims=2)

    # If there are any negative values in b, we can simply change the epigraph
    # variable to be that much higher to make the minimum 0. Note that this
    # won't work for constraints, it only works because it's for the objective
    # function and we have "z" in the tableau.
    # println("Step 1:")
    # display(Array(b_val))
    add_val = 0.0
    if any(<(0.0), b_val)
        add_val = -minimum(b_val)
        b_val .+= add_val
    end

    # Preemptively determine how many slack variables we'll need. This is going to be
    # the total number of cuts (an input to this function), plus one for the lower bound
    # value, plus the number of variables (since each has an upper bound of 1)
    slack_count = t.max_cuts+1+w # n_cuts, lower bound, w [upper bounds]
    
    # Create the stacked tableau as a big array of 0's, and then we fill it in as necessary.
    tableau = CUDA.zeros(Float64, l*(slack_count+1), 1+w+slack_count+1)
    solution_tableau = similar(tableau)

    # Fill in the first column, corresponding to the epigraph variable
    tableau[1:(slack_count+1):end,1] .= 1.0 # Lower bound constraint
    tableau[2:(slack_count+1):end,1] .= 1.0 # Only one cut to consider for now
    tableau[(slack_count+1):(slack_count+1):end,1] .= -1.0 # Objective row

    # Fill in the corrected subgradients in their respective columns
    tableau[2:(slack_count+1):end,2:1+w] .= corrected_subgradients

    # Fill in the slack variables (besides the rows for cuts we haven't done yet)
    for i = 1:slack_count
        if (i<=t.max_cuts+1) && (t.max_cuts>1) && (i>2)  #Reserving rows 3 to max_cuts+1 for future cuts
            continue
        end
        tableau[i:(slack_count+1):end,1+w+i] .= 1.0
    end

    # Fill in the lower bound for the epigraph variable 
    # (z >= lower_bound --> -z <= -lower_bound --> -z + v1 <= -lower_bound, and z is flipped to be -z
    tableau[1:(slack_count+1):end,end] .= @views -func_output[2][1:l] .+ add_val

    # Fill in the value for the first cut (b)
    tableau[2:(slack_count+1):end,end] .= b_val

    # Fill in the upper bounds for all the variables, which are all 1's
    for i = 1:w
        tableau[t.max_cuts+1+i:(slack_count+1):end,1+i] .= 1.0 # The variable itself
        tableau[t.max_cuts+1+i:(slack_count+1):end,end] .= 1.0 # The variable's upper bound (always 1 because we shifted it)
    end

    # Make sure all the rightmost column values are positive (not necessary, and for some
    # reason this eats up GPU memory?)
    # if any((@view tableau[:,end]) .< 0.0)
    #     error("Check b_val, might need an auxiliary system (or more creativity)")
    # end

    # Free up the func outputs since we no longer need them
    CUDA.unsafe_free!.(func_output)
    

    # Pass the tableau through the simplex algorithm and see what we get out of it
    # (Note that the solution values will be negated. That's fine, we don't actually
    # care what they are just yet.)
    solution_tableau .= tableau
    t.opt_time += @elapsed CUDA.@sync parallel_simplex(solution_tableau, slack_count+1)

    # Now we need to add another cut, which means we have to extract out the solution 
    # from the tableau, convert back into un-shifted variables, pass it back through
    # the convex evaluator, and add rows to the tableau.

    # Preallocate some arrays checks we'll be using 
    tableau_vals = CuArray{Float64}(undef, l,(slack_count+1))
    variable_vals = CuArray{Float64}(undef, l,(slack_count+1))
    bool_check = CuArray{Bool}(undef, l,(slack_count+1))
    zero_check = CuArray{Bool}(undef, l,(slack_count+1))
    for cut = 1:t.max_cuts-1 # Two additional cuts.
        # Extract solution values from the tableau to decide where the evaluations are.
        # Figure out which ones are "correct" for each variable?
        for i = 1:w
            tableau_vals .= reshape((@view solution_tableau[:,end]), (slack_count+1),l)'
            variable_vals .= reshape((@view solution_tableau[:,1+i]), (slack_count+1),l)'
            bool_check .= (variable_vals .== 1.0)
            zero_check .= (variable_vals .== 0.0)
            bool_check .&= (count(bool_check, dims=2).==1)
            bool_check .&= (count(zero_check, dims=2).==slack_count)
            tableau_vals .*= bool_check
            eval_points[:,i] .= @views min.(0.95, max.(0.05, sum(tableau_vals, dims=2))).*(uvbs_d[:,i] .- lvbs_d[:,i]).+lvbs_d[:,i]
        end

        # Okay, so now we have the points to evaluate, we need to call the function again
        t.relax_time += @elapsed CUDA.@sync func_output = @views t.convex_func_and_subgrad(
            ([[eval_points[:,i],eval_points[:,i],lvbs_d[:,i],uvbs_d[:,i]] for i=1:w]...)...) # n+2-dimensional

        # As before, calculate corrected subgradients and the new b_values
        subgradients .= hcat([func_output[i+2] for i in 1:w]...) #Only for the lower bound, not upper bound
        corrected_subgradients .= (subgradients).*(uvbs_d .- lvbs_d)
        mid_times_sub .= eval_points.*subgradients
        low_times_sub .= lvbs_d.*subgradients
        b_val .= sum(mid_times_sub, dims=2) .- func_output[1] .- sum(low_times_sub, dims=2)

        # Add in the extra factor
        b_val .+= add_val

        # If any of b_val is [still] negative, update add_val and the rest of the tableau
        if any(<(0.0), b_val)
            update = -minimum(b_val)
            b_val .+= update
            tableau[1:slack_count+1:end, end] .+= update
            for i = 2:(2+cut-1)
                tableau[i:slack_count+1:end, end] .+= update
            end
            add_val += update
        end

        # Free up the func outputs since we no longer need them
        CUDA.unsafe_free!.(func_output)

        # We can now place in these values into the tableau in the spots we left open earlier
        tableau[2+cut:slack_count+1:end, 1] .= 1.0
        tableau[2+cut:slack_count+1:end, 2:1+w] .= corrected_subgradients
        tableau[2+cut:slack_count+1:end, 1+w+2+cut] .= 1.0
        tableau[2+cut:slack_count+1:end, end] .= b_val

        # Adjust the final line of each problem to be the original minimization problem
        tableau[(slack_count+1):(slack_count+1):end,:] .= hcat(-CUDA.one(Float64), CUDA.zeros(Float64, 1, w+slack_count+1))

        # Run the simplex algorithm again
        solution_tableau .= tableau
        t.opt_time += @elapsed CUDA.@sync parallel_simplex(solution_tableau, slack_count+1)
    end

    # Save the lower bounds (remembering to negate the values)
    t.lower_bound_storage[1:t.node_len] .= @views Array(-(solution_tableau[slack_count+1:slack_count+1:end,end] .- add_val))

    for i in [lvbs_d, uvbs_d, eval_points, subgradients, corrected_subgradients, b_val,
              tableau, solution_tableau, tableau_vals, variable_vals, bool_check, zero_check]
        CUDA.unsafe_free!(i)
    end
    return nothing
end

function lower_and_upper_problem!(t::SimplexGPU_ObjAndCons, m::EAGO.GlobalOptimizer)
    ################################################################################
    ##########  Step 1) Determine problem parameters
    ################################################################################
    l, w = size(@view t.all_lvbs[1:t.node_len,:]) # points, num_vars
    geq_len = length(t.geq_cons)
    leq_len = length(t.leq_cons)
    eq_len = length(t.eq_cons)

    ################################################################################
    ##########  Step 2) Bring the bounds into the GPU
    ################################################################################
    lvbs_d = @views CuArray(t.all_lvbs[1:l,:])
    uvbs_d = @views CuArray(t.all_uvbs[1:l,:]) # [points x num_vars]

    
    ################################################################################
    ##########  Step 3) Preallocate space for mutating arguments of the objective 
    ##########          function and constraints
    ################################################################################
    # Objective function storage (2*l because we include upper bound calculations)
    obj_cv = CuArray{Float64}(undef, 2*l)
    obj_lo = CuArray{Float64}(undef, 2*l)
    obj_cvgrad =[CuArray{Float64}(undef, 2*l) for _ in 1:w]

    # GEQ constraint storage
    geq_cc = [CuArray{Float64}(undef, l) for _ in 1:geq_len]
    geq_ccgrad = [[CuArray{Float64}(undef, l) for _ in 1:w] for _ in 1:geq_len]
    
    # LEQ constraint storage
    leq_cv = [CuArray{Float64}(undef, l) for _ in 1:leq_len]
    leq_cvgrad = [[CuArray{Float64}(undef, l) for _ in 1:w] for _ in 1:leq_len]
    
    # EQ constraint storage
    eq_cv = [CuArray{Float64}(undef, l) for _ in 1:eq_len]
    eq_cc = [CuArray{Float64}(undef, l) for _ in 1:eq_len]
    eq_cvgrad = [[CuArray{Float64}(undef, l) for _ in 1:w] for _ in 1:eq_len]
    eq_ccgrad = [[CuArray{Float64}(undef, l) for _ in 1:w] for _ in 1:eq_len]
    
    
    ################################################################################
    ##########  Step 4) Set up points to evaluate (i.e. the centers of every node)
    ################################################################################
    eval_points = (lvbs_d .+ uvbs_d)./2


    ################################################################################
    ##########  Step 5) Calculate all required relaxations
    ################################################################################
    # Objective function (first 1:l are for the lower bound, l+1:end are for upper bound. This
    #                     is faster than calling the objective function twice due to GPU allocations in the function)
    @views t.obj_fun(obj_cv, obj_lo, obj_cvgrad..., 
                   ([[[eval_points[:,i];eval_points[:,i]],[eval_points[:,i];eval_points[:,i]],
                     [lvbs_d[:,i];eval_points[:,i]],[uvbs_d[:,i];eval_points[:,i]]] for i=1:w]...)...)

    # LEQ constraints
    for i in 1:leq_len
        @views t.leq_cons[i](leq_cv[i], leq_cvgrad[i]...,
                          ([[eval_points[:,j], eval_points[:,j], lvbs_d[:,j], uvbs_d[:,j]] for j=1:w]...)...)
    end
    
    # GEQ constraints
    for i in 1:geq_len
        @views t.geq_cons[i](geq_cc[i], geq_ccgrad[i]...,
                          ([[eval_points[:,j], eval_points[:,j], lvbs_d[:,j], uvbs_d[:,j]] for j=1:w]...)...)
    end
    
    # EQ constraints
    for i in 1:eq_len
        @views t.eq_cons[i](eq_cv[i], eq_cc[i], eq_cvgrad[i]..., eq_ccgrad[i]...,
                          ([[eval_points[:,j], eval_points[:,j], lvbs_d[:,j], uvbs_d[:,j]] for j=1:w]...)...)
    end

    # Store the upper bounds
    t.upper_bound_storage[1:l] .= @views Array(obj_cv[l+1:end])


    ################################################################################
    ##########  Step 6) Create the stacked Simplex tableau
    ################################################################################
    # We can start by creating the basic tableau skeleton, which doesn't require
    # any of the calculated information. First, determine how many extra columns
    # are needed for slack variables. We need one for each variable's upper bound 
    # (which will be scaled to 1), one for the lower bound of the objective function, 
    # and then for each cut we'll need one for the objective function and one for
    # each constraint
    slack_count = Int32(w + 1 + (t.max_cuts * (1 + geq_len + leq_len + 2*eq_len)))

    # In addition to slack variables, there are artificial variables. Artificial
    # variables get used if the rightmost column (b) is negative, or if the 
    # constraint is a GEQ constraint (but not both of these, or the negatives
    # cancel out). However, because all variables are scaled to [0, 1], we will
    # never need an artificial variable for the "w" variable rows. Hence, there
    # are (slack_count - w) artificial variable rows. This makes the total width
    # of the tableau equal to 2 for the epigraph variables, plus the number of 
    # variables "w", plus the slack variables (slack_count), plus the number of 
    # artificial variables (slack_count - w), plus 1 for the final column. 
    # The w's cancel, and we get (2*slack_count + 3)
    tableau_width = Int32(2*slack_count + 3)

    # Allocate space for the tableau and working tableau, and create the basic
    # stacked Simplex tableau skeleton
    tableau = CUDA.zeros(Float64, l*(slack_count+2), tableau_width)
    working_tableau = similar(tableau)
    CUDA.@cuda blocks=l threads=768 tableau_skeleton_kernel(tableau, w, slack_count)

    # Add in a "cut" that is the lower bound of the objective function
    CUDA.@cuda blocks=l threads=640 add_lower_bound_kernel(tableau, (@view obj_lo[1:l]), w+Int32(1), slack_count+Int32(2), l)


    ################################################################################
    ##########  Step 7) Add cut information
    ################################################################################

    # Now that the skeleton is in place, we can add individual rows for each of
    # the calculated subgradients, for the objective and constraints
    subgradients = CuArray{Float64}(undef, l, w)
    scaled_subgradients = CuArray{Float64}(undef, l, w)
    hyperplane_mid = CuArray{Float64}(undef, l)
    hyperplane_low = CuArray{Float64}(undef, l)
    b_val = CuArray{Float64}(undef, l)
    thread_count = Int32(2^floor(Int, log2(w-1))) # Active threads in the rowsum operation

    # Start with the objective. Note that we must scale the subtangent hyperplanes
    # to be on [0, 1] instead of the original domains. Given a hyperplane of:
    # m*x + n*y = c, on [xL, xU], [yL, yU]
    # We can scale to [0, 1], [0, 1] by shifting the intercept from (0, 0) to (xL, yL),
    # and then multiplying the subgradient terms by (xU - xL) and (yU - yL), respectively.
    # The subgradients m and n come directly from subgradient calculations, and c
    # can be calculated since we know that m*(eval_x) + n*(eval_y) - c = (cv or cc)
    # based on the point of evaluation (eval_x, eval_y) and the value of the relaxation
    # (cv or cc), depending on which type of relaxation was calculated. This value
    # can be scaled to (0,0) from (xL, yL) by subtracting the slopes times the lower
    # bounds. 
    # That is, we can calculate:
    # m' = m*(xU - xL)
    # n' = n*(yU - yL)
    # c' = m*(eval_x) + n*(eval_y) - (m*(xL) + n*(yL)) - (cv or cc)
    # To get the final hyperplane:
    # m'*x + n'*y = c'
    subgradients .= @views hcat([obj_cvgrad[i][1:l] for i in 1:w]...)
    scaled_subgradients .= subgradients .* (uvbs_d .- lvbs_d)
    CUDA.@cuda blocks=l threads=thread_count shmem=8*w accumulate_mul_kernel(hyperplane_mid, subgradients, eval_points)
    CUDA.@cuda blocks=l threads=thread_count shmem=8*w accumulate_mul_kernel(hyperplane_low, subgradients, lvbs_d)
    b_val .= hyperplane_mid .- hyperplane_low .- (@view obj_cv[1:l])

    # Now we can add in the objective cut for the l problems.
    CUDA.@cuda threads=512 add_cut_kernel(tableau, scaled_subgradients, b_val, Int32(w+2), w, slack_count+Int32(2), l, slack_count, true, false)

    # Now we can do a similar process for the constraints
    for i = 1:leq_len
        subgradients .= @views hcat([leq_cvgrad[i][j] for j = 1:w]...)
        scaled_subgradients .= subgradients .* (uvbs_d .- lvbs_d)
        CUDA.@cuda blocks=l threads=thread_count shmem=8*w accumulate_mul_kernel(hyperplane_mid, subgradients, eval_points)
        CUDA.@cuda blocks=l threads=thread_count shmem=8*w accumulate_mul_kernel(hyperplane_low, subgradients, lvbs_d)
        b_val .= hyperplane_mid .- hyperplane_low .- leq_cv[i]
        CUDA.@cuda threads=512 add_cut_kernel(tableau, scaled_subgradients, b_val, Int32(w+2+i), w, slack_count+Int32(2), l, slack_count, false, false)
    end
    for i = 1:geq_len
        subgradients .= @views hcat([geq_ccgrad[i][j] for j = 1:w]...)
        scaled_subgradients .= subgradients .* (uvbs_d .- lvbs_d)
        CUDA.@cuda blocks=l threads=thread_count shmem=8*w accumulate_mul_kernel(hyperplane_mid, subgradients, eval_points)
        CUDA.@cuda blocks=l threads=thread_count shmem=8*w accumulate_mul_kernel(hyperplane_low, subgradients, lvbs_d)
        b_val .= hyperplane_mid .- hyperplane_low .- geq_cc[i]
        CUDA.@cuda threads=512 add_cut_kernel(tableau, scaled_subgradients, b_val, Int32(w+2+leq_len+i), w, slack_count+Int32(2), l, slack_count, false, true)
    end
    for i = 1:eq_len
        # Repeat what happened for LEQ and GEQ constraints, but EQ constraints have both.
        subgradients .= @views hcat([eq_cvgrad[i][j] for j = 1:w]...)
        scaled_subgradients .= subgradients .* (uvbs_d .- lvbs_d)
        CUDA.@cuda blocks=l threads=thread_count shmem=8*w accumulate_mul_kernel(hyperplane_mid, subgradients, eval_points)
        CUDA.@cuda blocks=l threads=thread_count shmem=8*w accumulate_mul_kernel(hyperplane_low, subgradients, lvbs_d)
        b_val .= hyperplane_mid .- hyperplane_low .- eq_cv[i]
        CUDA.@cuda threads=512 add_cut_kernel(tableau, scaled_subgradients, b_val, Int32(w+2+leq_len+geq_len+(2*i-1)), w, slack_count+Int32(2), l, slack_count, false, false)
        subgradients .= @views hcat([eq_ccgrad[i][j] for j = 1:w]...)
        scaled_subgradients .= subgradients .* (uvbs_d .- lvbs_d)
        CUDA.@cuda blocks=l threads=thread_count shmem=8*w accumulate_mul_kernel(hyperplane_mid, subgradients, eval_points)
        CUDA.@cuda blocks=l threads=thread_count shmem=8*w accumulate_mul_kernel(hyperplane_low, subgradients, lvbs_d)
        b_val .= hyperplane_mid .- hyperplane_low .- eq_cc[i]
        CUDA.@cuda threads=512 add_cut_kernel(tableau, scaled_subgradients, b_val, Int32(w+2+leq_len+geq_len+(2*i)), w, slack_count+Int32(2), l, slack_count, false, true)
    end

    ################################################################################
    ##########  Step 8) Run the Simplex algorithm
    ################################################################################

    # Pass the tableau through the simplex algorithm and see what we get out of it
    # (Note that the solution values will be negated. That's fine, we don't actually
    # care what they are just yet.)
    # device_synchronize()
    working_tableau .= tableau
    # display(Array(tableau))
    twophase_parallel_simplex(working_tableau, w, slack_count+2)
    # display(Array(working_tableau))
    # Note that the solutions are on lines [slack_count+1 : slack_count+2 : end, end]

    ################################################################################
    ##########  Step 9) Add additional cuts if necessary
    ################################################################################

    # Now we need to add another cut, which means we have to extract out the solution 
    # from the tableau, convert back into un-shifted variables, pass it back through
    # the convex evaluator, and add rows to the tableau.

    # If we're doing any more cuts...
    if t.max_cuts > 1
        # Remake storage for the objective function, now that we don't need upper bound
        # calculations anymore 
        CUDA.unsafe_free!(obj_cv)
        CUDA.unsafe_free!(obj_lo)
        CUDA.unsafe_free!.(obj_cvgrad)
        obj_cv = CuArray{Float64}(undef, l)
        obj_lo = CuArray{Float64}(undef, l)
        obj_cvgrad =[CuArray{Float64}(undef, l) for _ in 1:w]

        # Preallocate some arrays checks we'll be using 
        tableau_vals = CuArray{Float64}(undef, l,(slack_count+2))
        variable_vals = CuArray{Float64}(undef, l,(slack_count+2))
        bool_check = CuArray{Bool}(undef, l,(slack_count+2))
        zero_check = CuArray{Bool}(undef, l,(slack_count+2))
        for cut = 1:t.max_cuts-1
            # Extract solution values from the tableau to decide where the evaluations are.
            # Figure out which ones are "correct" for each variable?
            for i = 1:w
                tableau_vals .= reshape((@view working_tableau[:,end]), (slack_count+2),l)'
                variable_vals .= reshape((@view working_tableau[:,2+i]), (slack_count+2),l)'
                bool_check .= (variable_vals .== 1.0)
                zero_check .= (variable_vals .== 0.0)
                bool_check .&= (count(bool_check, dims=2).==1)
                bool_check .&= (count(zero_check, dims=2).==slack_count+1)
                tableau_vals .*= bool_check
                eval_points[:,i] .= @views min.(0.95, max.(0.05, sum(tableau_vals, dims=2))).*(uvbs_d[:,i] .- lvbs_d[:,i]).+lvbs_d[:,i]
            end

            # Okay, so now we have the points to evaluate, we need to call all the functions again
            # Objective function
            @views t.obj_fun(obj_cv, obj_lo, obj_cvgrad..., 
                        ([[eval_points[:,i],eval_points[:,i],lvbs_d[:,i],uvbs_d[:,i]] for i=1:w]...)...)

            # LEQ constraints
            for i in 1:leq_len
                @views t.leq_cons[i](leq_cv[i], leq_cvgrad[i]...,
                                ([[eval_points[:,j], eval_points[:,j], lvbs_d[:,j], uvbs_d[:,j]] for j=1:w]...)...)
            end
            
            # GEQ constraints
            for i in 1:geq_len
                @views t.geq_cons[i](geq_cc[i], geq_ccgrad[i]...,
                                ([[eval_points[:,j], eval_points[:,j], lvbs_d[:,j], uvbs_d[:,j]] for j=1:w]...)...)
            end
            
            # EQ constraints
            for i in 1:eq_len
                @views t.eq_cons[i](eq_cv[i], eq_cc[i], eq_cvgrad[i]..., eq_ccgrad[i]...,
                                ([[eval_points[:,j], eval_points[:,j], lvbs_d[:,j], uvbs_d[:,j]] for j=1:w]...)...)
            end

            # And, as before, we can add the cuts to the tableau one by one
            subgradients .= @views hcat([obj_cvgrad[i] for i in 1:w]...)
            scaled_subgradients .= subgradients .* (uvbs_d .- lvbs_d)
            CUDA.@cuda blocks=l threads=thread_count shmem=8*w accumulate_mul_kernel(hyperplane_mid, subgradients, eval_points)
            CUDA.@cuda blocks=l threads=thread_count shmem=8*w accumulate_mul_kernel(hyperplane_low, subgradients, lvbs_d)
            b_val .= hyperplane_mid .- hyperplane_low .- obj_cv

            # Now we can add in the objective cut for the l problems.
            # Note that the row we're adding to is moved forward depending on which
            # cut we're on.
            shift = Int32(cut * (1 + geq_len + leq_len + 2*eq_len))
            CUDA.@cuda threads=512 add_cut_kernel(tableau, scaled_subgradients, b_val, Int32(shift+w+2), w, slack_count+Int32(2), l, slack_count, true, false)

            # Now we can do a similar process for the constraints
            for i = 1:leq_len
                subgradients .= @views hcat([leq_cvgrad[i][j] for j = 1:w]...)
                scaled_subgradients .= subgradients .* (uvbs_d .- lvbs_d)
                CUDA.@cuda blocks=l threads=thread_count shmem=8*w accumulate_mul_kernel(hyperplane_mid, subgradients, eval_points)
                CUDA.@cuda blocks=l threads=thread_count shmem=8*w accumulate_mul_kernel(hyperplane_low, subgradients, lvbs_d)
                b_val .= hyperplane_mid .- hyperplane_low .- leq_cv[i]
                CUDA.@cuda threads=512 add_cut_kernel(tableau, scaled_subgradients, b_val, Int32(shift+w+2+i), w, slack_count+Int32(2), l, slack_count, false, false)
            end
            for i = 1:geq_len
                subgradients .= @views hcat([geq_ccgrad[i][j] for j = 1:w]...)
                scaled_subgradients .= subgradients .* (uvbs_d .- lvbs_d)
                CUDA.@cuda blocks=l threads=thread_count shmem=8*w accumulate_mul_kernel(hyperplane_mid, subgradients, eval_points)
                CUDA.@cuda blocks=l threads=thread_count shmem=8*w accumulate_mul_kernel(hyperplane_low, subgradients, lvbs_d)
                b_val .= hyperplane_mid .- hyperplane_low .- geq_cc[i]
                CUDA.@cuda threads=512 add_cut_kernel(tableau, scaled_subgradients, b_val, Int32(shift+w+2+leq_len+i), w, slack_count+Int32(2), l, slack_count, false, true)
            end
            for i = 1:eq_len
                # Repeat what happened for LEQ and GEQ constraints, but EQ constraints have both.
                subgradients .= @views hcat([eq_cvgrad[i][j] for j = 1:w]...)
                scaled_subgradients .= subgradients .* (uvbs_d .- lvbs_d)
                CUDA.@cuda blocks=l threads=thread_count shmem=8*w accumulate_mul_kernel(hyperplane_mid, subgradients, eval_points)
                CUDA.@cuda blocks=l threads=thread_count shmem=8*w accumulate_mul_kernel(hyperplane_low, subgradients, lvbs_d)
                b_val .= hyperplane_mid .- hyperplane_low .- eq_cv[i]
                CUDA.@cuda threads=512 add_cut_kernel(tableau, scaled_subgradients, b_val, Int32(shift+w+2+leq_len+geq_len+(2*i-1)), w, slack_count+Int32(2), l, slack_count, false, false)
                subgradients .= @views hcat([eq_ccgrad[i][j] for j = 1:w]...)
                scaled_subgradients .= subgradients .* (uvbs_d .- lvbs_d)
                CUDA.@cuda blocks=l threads=thread_count shmem=8*w accumulate_mul_kernel(hyperplane_mid, subgradients, eval_points)
                CUDA.@cuda blocks=l threads=thread_count shmem=8*w accumulate_mul_kernel(hyperplane_low, subgradients, lvbs_d)
                b_val .= hyperplane_mid .- hyperplane_low .- eq_cc[i]
                CUDA.@cuda threads=512 add_cut_kernel(tableau, scaled_subgradients, b_val, Int32(shift+w+2+leq_len+geq_len+(2*i)), w, slack_count+Int32(2), l, slack_count, false, true)
            end

            # Run the simplex algorithm again
            working_tableau .= tableau
            twophase_parallel_simplex(working_tableau, w, slack_count+2)
        end
        for i in [tableau_vals, variable_vals, bool_check, zero_check]
            CUDA.unsafe_free!(i)
        end
    end

    # Save the lower bounds (Note that it's the second from the bottom row in each LP,
    # and we're remembering to negate the values)
    t.lower_bound_storage[1:l] .= @views Array(-(working_tableau[slack_count+1:slack_count+2:end,end]))

    for i in [lvbs_d, uvbs_d, eval_points, obj_cv, obj_lo, obj_cvgrad, geq_cc, geq_ccgrad,
              leq_cv, leq_cvgrad, eq_cv, eq_cc, eq_cvgrad, eq_ccgrad, subgradients, 
              scaled_subgradients, hyperplane_mid, hyperplane_low, b_val, tableau, 
              working_tableau]
        if typeof(i) <: Vector
            if !isempty(i)
                if typeof(i[1]) <: Vector
                    for j in i
                        CUDA.unsafe_free!.(j)
                    end
                else
                    CUDA.unsafe_free!.(i)
                end
            end
        else
            CUDA.unsafe_free!(i)
        end
    end
    return nothing
end

function lower_and_upper_problem!(t::SimplexGPU_ObjOnly_Mat, m::EAGO.GlobalOptimizer)
    ################################################################################
    ##########  Step 1) Determine problem parameters
    ################################################################################
    l, w = size(@view t.all_lvbs[1:t.node_len,:]) # points, num_vars
    t.lower_counter += 1
    t.node_counter += t.node_len

    ################################################################################
    ##########  Step 2) Bring the bounds into the GPU
    ################################################################################
    lvbs_d = @views CuArray(t.all_lvbs[1:l,:])
    uvbs_d = @views CuArray(t.all_uvbs[1:l,:]) # [points x num_vars]

    
    ################################################################################
    ##########  Step 3) Preallocate space for mutating arguments of the objective 
    ##########          function and constraints
    ################################################################################
    # Objective function storage (2*l because we include upper bound calculations)
    obj_cv = CuArray{Float64}(undef, 2*l)
    obj_lo = CuArray{Float64}(undef, 2*l)
    obj_cvgrad = CuArray{Float64}(undef, 2*l, w)

    
    ################################################################################
    ##########  Step 4) Set up points to evaluate (i.e. the centers of every node)
    ################################################################################
    eval_points = (lvbs_d .+ uvbs_d)./2


    ################################################################################
    ##########  Step 5) Calculate all required relaxations
    ################################################################################
    # Objective function (first 1:l are for the lower bound, l+1:end are for upper bound. This
    #                     is faster than calling the objective function twice due to GPU allocations in the function)
    # start = t.relax_time
    CUDA.NVTX.@range "Relaxations" begin
    CUDA.@profile t.relax_time += @elapsed @views t.obj_fun(obj_cv, obj_lo, obj_cvgrad, 
    # @views t.obj_fun(obj_cv, obj_lo, obj_cvgrad, 
                   ([[[eval_points[:,i];eval_points[:,i]],[eval_points[:,i];eval_points[:,i]],
                     [lvbs_d[:,i];eval_points[:,i]],[uvbs_d[:,i];eval_points[:,i]]] for i=1:w]...)...)
    end
    # @show t.node_len
    # @show t.relax_time-start
    error()

    # Store the upper bounds
    t.upper_bound_storage[1:l] .= @views Array(obj_cv[l+1:end])


    ################################################################################
    ##########  Step 6) Create the stacked Simplex tableau
    ################################################################################
    # We can start by creating the basic tableau skeleton, which doesn't require
    # any of the calculated information. First, determine how many extra columns
    # are needed for slack variables. We need one for each variable's upper bound 
    # (which will be scaled to 1), one for the lower bound of the objective function, 
    # and then for each cut we'll need one for the objective function and one for
    # each constraint
    slack_count = Int32(w + 1 + t.max_cuts)

    # In addition to slack variables, there are artificial variables. Artificial
    # variables get used if the rightmost column (b) is negative, or if the 
    # constraint is a GEQ constraint (but not both of these, or the negatives
    # cancel out). However, because all variables are scaled to [0, 1], we will
    # never need an artificial variable for the "w" variable rows. Hence, there
    # are (slack_count - w) artificial variable rows. This makes the total width
    # of the tableau equal to 2 for the epigraph variables, plus the number of 
    # variables "w", plus the slack variables (slack_count), plus the number of 
    # artificial variables (slack_count - w), plus 1 for the final column. 
    # The w's cancel, and we get (2*slack_count + 3)
    tableau_width = Int32(2*slack_count + 3)

    # Allocate space for the tableau and working tableau, and create the basic
    # stacked Simplex tableau skeleton
    tableau = CUDA.zeros(Float64, l*(slack_count+2), tableau_width)
    working_tableau = similar(tableau)
    CUDA.@cuda blocks=l threads=768 tableau_skeleton_kernel(tableau, w, slack_count)

    # Add in a "cut" that is the lower bound of the objective function
    CUDA.@cuda blocks=l threads=640 add_lower_bound_kernel(tableau, (@view obj_lo[1:l]), w+Int32(1), slack_count+Int32(2), l)


    ################################################################################
    ##########  Step 7) Add cut information
    ################################################################################

    # Now that the skeleton is in place, we can add individual rows for each of
    # the calculated subgradients, for the objective and constraints
    subgradients = CuArray{Float64}(undef, l, w)
    scaled_subgradients = CuArray{Float64}(undef, l, w)
    hyperplane_mid = CuArray{Float64}(undef, l)
    hyperplane_low = CuArray{Float64}(undef, l)
    b_val = CuArray{Float64}(undef, l)
    thread_count = Int32(2^floor(Int, log2(w-1))) # Active threads in the rowsum operation
    degeneracy_flag = CUDA.zeros(Bool, l)
    blocks = Int32(min(cld(l,512),1024))

    # Start with the objective. Note that we must scale the subtangent hyperplanes
    # to be on [0, 1] instead of the original domains. Given a hyperplane of:
    # m*x + n*y = c, on [xL, xU], [yL, yU]
    # We can scale to [0, 1], [0, 1] by shifting the intercept from (0, 0) to (xL, yL),
    # and then multiplying the subgradient terms by (xU - xL) and (yU - yL), respectively.
    # The subgradients m and n come directly from subgradient calculations, and c
    # can be calculated since we know that m*(eval_x) + n*(eval_y) - c = (cv or cc)
    # based on the point of evaluation (eval_x, eval_y) and the value of the relaxation
    # (cv or cc), depending on which type of relaxation was calculated. This value
    # can be scaled to (0,0) from (xL, yL) by subtracting the slopes times the lower
    # bounds. 
    # That is, we can calculate:
    # m' = m*(xU - xL)
    # n' = n*(yU - yL)
    # c' = m*(eval_x) + n*(eval_y) - (m*(xL) + n*(yL)) - (cv or cc)
    # To get the final hyperplane:
    # m'*x + n'*y = c'
    subgradients .= @view obj_cvgrad[1:l,:]
    scaled_subgradients .= subgradients .* (uvbs_d .- lvbs_d)
    CUDA.@cuda blocks=l threads=thread_count shmem=8*w accumulate_mul_kernel(hyperplane_mid, subgradients, eval_points)
    CUDA.@cuda blocks=l threads=thread_count shmem=8*w accumulate_mul_kernel(hyperplane_low, subgradients, lvbs_d)
    b_val .= hyperplane_mid .- hyperplane_low .- (@view obj_cv[1:l])

    # Now we can add in the objective cut for the l problems.
    CUDA.@cuda blocks=blocks threads=512 add_cut_kernel(tableau, scaled_subgradients, b_val, Int32(w+2), w, slack_count+Int32(2), l, slack_count, true, false, degeneracy_flag)

    # println("1024 and 1025 for cut: 1")
    # display(Array(tableau)[1023*(slack_count+2)+1 : 1025*(slack_count+2),:])

    ################################################################################
    ##########  Step 8) Run the Simplex algorithm
    ################################################################################

    # Pass the tableau through the simplex algorithm and see what we get out of it
    # (Note that the solution values will be negated. That's fine, we don't actually
    # care what they are just yet.)
    # device_synchronize()
    working_tableau .= tableau
    t.opt_time += @elapsed twophase_parallel_simplex(working_tableau, w, slack_count+2)
    # twophase_parallel_simplex(working_tableau, w, slack_count+2)
    # Note that the solutions are on lines [slack_count+1 : slack_count+2 : end, end]
    # error()

    ################################################################################
    ##########  Step 9) Add additional cuts if necessary
    ################################################################################

    # Now we need to add another cut, which means we have to extract out the solution 
    # from the tableau, convert back into un-shifted variables, pass it back through
    # the convex evaluator, and add rows to the tableau.

    # If we're doing any more cuts...
    if t.max_cuts > 1
        # Remake storage for the objective function, now that we don't need upper bound
        # calculations anymore 
        CUDA.unsafe_free!(obj_cv)
        CUDA.unsafe_free!(obj_lo)
        CUDA.unsafe_free!(obj_cvgrad)
        obj_cv = CuArray{Float64}(undef, l)
        obj_lo = CuArray{Float64}(undef, l)
        obj_cvgrad = CuArray{Float64}(undef, l, w)
        previous_sol = CuArray{Float64}(undef, l, w)

        # Preallocate some arrays checks we'll be using 
        tableau_vals = CuArray{Float64}(undef, l,(slack_count+2))
        variable_vals = CuArray{Float64}(undef, l,(slack_count+2))
        bool_check = CuArray{Bool}(undef, l,(slack_count+2))
        zero_check = CuArray{Bool}(undef, l,(slack_count+2))
        for cut = 1:t.max_cuts-1
            # Extract solution values from the tableau to decide where the evaluations are.
            # Figure out which ones are "correct" for each variable
            for i = 1:w
                tableau_vals .= reshape((@view working_tableau[:,end]), (slack_count+2),l)'
                variable_vals .= reshape((@view working_tableau[:,2+i]), (slack_count+2),l)'
                bool_check .= (variable_vals .== 1.0)
                zero_check .= (variable_vals .== 0.0)
                bool_check .&= (count(bool_check, dims=2).==1)
                bool_check .&= (count(zero_check, dims=2).==slack_count+1)
                tableau_vals .*= bool_check
                eval_points[:,i] .= @views min.(0.95, max.(0.05, sum(tableau_vals, dims=2))).*(uvbs_d[:,i] .- lvbs_d[:,i]).+lvbs_d[:,i]
            end

            # Run a degeneracy check to compare eval_points against previous_sol.
            # If there's a row where they're exactly equal, we don't add that cut.
            CUDA.@cuda blocks=blocks threads=1024 degeneracy_check_kernel(degeneracy_flag, previous_sol, eval_points)

            # Set the previous solution matrix to be the current solutions
            previous_sol .= eval_points

            # Okay, so now we have the points to evaluate, we need to call all the functions again
            # Objective function
            t.relax_time += @elapsed @views t.obj_fun(obj_cv, obj_lo, obj_cvgrad, 
            # @views t.obj_fun(obj_cv, obj_lo, obj_cvgrad, 
                        ([[eval_points[:,i],eval_points[:,i],lvbs_d[:,i],uvbs_d[:,i]] for i=1:w]...)...)

            # And, as before, we can add the cuts to the tableau one by one
            subgradients .= obj_cvgrad
            scaled_subgradients .= subgradients .* (uvbs_d .- lvbs_d)
            CUDA.@cuda blocks=l threads=thread_count shmem=8*w accumulate_mul_kernel(hyperplane_mid, subgradients, eval_points)
            CUDA.@cuda blocks=l threads=thread_count shmem=8*w accumulate_mul_kernel(hyperplane_low, subgradients, lvbs_d)
            b_val .= hyperplane_mid .- hyperplane_low .- obj_cv

            # Now we can add in the objective cut for the l problems.
            # Note that the row we're adding to is moved forward depending on which
            # cut we're on.
            shift = Int32(cut)
            CUDA.@cuda blocks=blocks threads=512 add_cut_kernel(tableau, scaled_subgradients, b_val, Int32(shift+w+2), w, slack_count+Int32(2), l, slack_count, true, false, degeneracy_flag)

            # Run the simplex algorithm again
            working_tableau .= tableau
            t.opt_time += @elapsed twophase_parallel_simplex(working_tableau, w, slack_count+2)
            # twophase_parallel_simplex(working_tableau, w, slack_count+2)
        end
        for i in [tableau_vals, variable_vals, bool_check, zero_check]
            CUDA.unsafe_free!(i)
        end
    end

    # Save the lower bounds (Note that it's the second from the bottom row in each LP,
    # and we're remembering to negate the values)
    t.lower_bound_storage[1:l] .= @views Array(-(working_tableau[slack_count+1:slack_count+2:end,end]))

    for i in [lvbs_d, uvbs_d, eval_points, obj_cv, obj_lo, obj_cvgrad, subgradients, 
              scaled_subgradients, hyperplane_mid, hyperplane_low, b_val, tableau, 
              working_tableau]
        if typeof(i) <: Vector
            if !isempty(i)
                if typeof(i[1]) <: Vector
                    for j in i
                        CUDA.unsafe_free!.(j)
                    end
                else
                    CUDA.unsafe_free!.(i)
                end
            end
        else
            CUDA.unsafe_free!(i)
        end
    end
    return nothing
end

# A fourth version of lower_and_upper_problem! that uses the new GPU Simplex algorithm
# but that only checks the midpoint of the node to get subgradients
function lower_and_upper_problem!(t::SimplexGPU_Single, m::EAGO.GlobalOptimizer)
    # Step 1) Bring the bounds into the GPU
    lvbs_d = CuArray(t.all_lvbs)
    uvbs_d = CuArray(t.all_uvbs) # [points x num_vars]

    # Step 2) Preallocate points to evaluate 
    l, w = size(t.all_lvbs) #points, num_vars
    np = 2 #Number of points; Center point, and one extra for upper bound calculations
    eval_points = Vector{CuArray{Float64}}(undef, 3*w) #Only 3x because one is repeated (cv or cc, lo, hi)
    for i = 1:w
        eval_points[3i-2] = CuArray{Float64}(undef, l*np)
        eval_points[3i-1] = repeat(lvbs_d[:,i], inner=np)
        eval_points[3i] = repeat(uvbs_d[:,i], inner=np)
    end
    bounds_d = CuArray{Float64}(undef, l)
    
    # Step 3) Fill in the variable midpoints for each node
    for i = 1:w
        # Each variable's cv (or cc) value is the midpoint of the node
        # (lower and upper bounds specified previously)
        eval_points[3i-2][1:np:end] .= (lvbs_d[:,i].+uvbs_d[:,i])./2

        # Now we do np:np:end. Each one is set to the center of the variable bounds,
        # creating a degenerate interval. This gives us the upper bound for the node.
        eval_points[3i-2][np:np:end] .= (lvbs_d[:,i].+uvbs_d[:,i])./2
        eval_points[3i-1][np:np:end] .= (lvbs_d[:,i].+uvbs_d[:,i])./2
        eval_points[3i][np:np:end] .= (lvbs_d[:,i].+uvbs_d[:,i])./2
    end

    # Step 4) Prepare the input vector for the convex function
    input = Vector{CuArray{Float64}}(undef, 0)
    for i = 1:w
        push!(input, [eval_points[3i-2], eval_points[3i-2], eval_points[3i-1], eval_points[3i]]...)
    end

    # Step 5) Perform the calculations
    func_output = t.convex_func_and_subgrad(input...) # n+2-dimensional
    # Also need whatever constraints!!

    # Step 6) Use values and subgradients to prepare the stacked Simplex tableau

    # First things first, we can prepare the "b" vector and see if we need any auxiliary systems.
    # This step calculates the intercept of b at x=x_lo, which is equivalent to calculating
    # the intercept at x=0 and then later shifting x_lo to 0, but without the extra re-calculation
    # steps
    b_start = func_output[1]
    for i = 1:w
        b_start -= func_output[i+2].*(eval_points[3i-2] .- eval_points[3i-1])

        # func_output[i+2] is the subgradient of the convex relaxation in the i'th dimension
        # eval_points[3i-2] is the cv/cc point used to obtain the relaxation
        # eval_points[3i-1] is the lower bound for this relaxation
        # eval_points[3i] is the upper bound (which isn't used here)

        # Note that <= [upper bound] will change to <= [upper bound] - [lower bound] 
        # for each variable, later
    end

    if all(<=(0.0), b_start) 
        #If b_start is all nonpositive, we don't need any auxiliary systems

        # Start making the tableau as normal, since we have a basic feasible solution at the start.
        # Create an extended b_array 
        b_array = vcat([vcat(-b_start[np*(j-1)+1:np*j-1],   # First "1:(np-1)" points for each node
                             uvbs_d[j,:].-lvbs_d[j,:],      # Upper bound minus lower bound 
                             0.0)                           # 0.0 for the objective function row
                        for j=1:l]...) # Repeat for every node
        
        # Prepare the epigraph variable columns. We're minimizing "Z", but since "Z" is unbounded, we
        # convert it to Z = Z_pos - Z_neg, where Z_pos, Z_neg >= 0.0. The first column will be Z_pos,
        # and the second column will be Z_neg. The upper bound rows and auxiliary system objective
        # function row will have these as 0; the objective function row will be [1, -1] (minimizing
        # Z_pos - Z_neg); and the constraints associated with Z will be [-1, 1] (-Z = -Z_pos + Z_neg)
        epigraph = hcat(-CUDA.ones(Float64, length(b_array)), CUDA.ones(Float64, length(b_array)))
        for i = 1:w
            # Starting at the first upper bound, repeat for every tableau
            epigraph[np+i-1 : np+w : end, :] .= 0.0
        end
        epigraph[np+w : np+w : end, :] .*= -1.0 # The main objective function is opposite the other rows (minimizing Z)
        epigraph[np+w+1 : np+w : end, :] .= 0.0 # The epigraph column is 0 in the auxiliary objective row


        # Combine the epigraph columns, "A" matrix, slack variable columns, and "b" array into the stacked tableaus
        tableaus = hcat(epigraph,                                                       # Epigraph variable columns
                        [vcat([vcat(func_output[i+2][np*(j-1)+1:np*j-1],                    # >>Subgradient values for the i'th variable for the j'th node
                            CUDA.zeros(Float64, w),                                         # >>Zeros for upper bound constraints (will fill in later with 1.0s)
                            0.0)                                                            # >>0.0 for the objective function row
                            for j = 1:l]...)                                                # Repeat for every j'th node vertically
                        for i = 1:w]...,                                                # Add a column for every i'th variable
                        [CUDA.zeros(Float64, length(b_array)) for _ = 1:(np-1)+w]...,   # Slack variables (will fill in later with 1.0s)
                        b_array)                                                        # The array of b's
        
        # Fill in the upper bound constraint indices and the slack variables
        for i = 1:w
            tableaus[np+i-1 : np+w : end, i+2] .= 1.0
        end
        for i = 1:(np-1)+w
            tableaus[i:np+w:end, (w+2)+i] .= 1.0
        end

        tableaus .= parallel_simplex(tableaus, np+w)

    else
        # It was detected that some slack variable coefficient would be negative, so we need to make auxiliary systems. 
        # Note: It's probably worth it to only do auxiliary systems for the tableaus that will need it. At least check 
        # to see how common this is, and whether it'll be necessary...

        # Create the extended b_array
        b_array = vcat([vcat(-b_start[np*(j-1)+1:np*j-1],   # First "1:(np-1)" points for each node
                             uvbs_d[j,:].-lvbs_d[j,:],      # Upper bound minus lower bound 
                             0.0,                           # 0.0 for the objective function row
                             0.0)                           # 0.0 for the auxiliary system objective function row
                        for j=1:l]...) # Repeat for every node

        # (NOTE: Should we be scaling all the variables/subgradients so that the variables are bounded on [0, 1]?)

        # Prepare the epigraph variable columns. We're minimizing "Z", but since "Z" is unbounded, we
        # convert it to Z = Z_pos - Z_neg, where Z_pos, Z_neg >= 0.0. The first column will be Z_pos,
        # and the second column will be Z_neg. The upper bound rows and auxiliary system objective
        # function row will have these as 0; the objective function row will be [1, -1] (minimizing
        # Z_pos - Z_neg); and the constraints associated with Z will be [-1, 1] (-Z = -Z_pos + Z_neg)
        epigraph = hcat(-CUDA.ones(Float64, length(b_array)), CUDA.ones(Float64, length(b_array)))
        for i = 1:w
            # Starting at the first upper bound, repeat for every tableau 
            # (which has np+w+1 rows thanks to the auxiliary row)
            epigraph[np+i-1 : np+w+1 : end, :] .= 0.0 
        end
        epigraph[np+w : np+w+1 : end, :] .*= -1.0 # The main objective function is opposite the other rows (minimizing Z)
        epigraph[np+w+1 : np+w+1 : end, :] .= 0.0 # The epigraph column is 0 in the auxiliary objective row

        # Combine the epigraph columns, "A" matrix, slack variable columns, and "b" array into the stacked tableaus
        tableaus = hcat(epigraph,                                                         # Epigraph variable columns
                        [vcat([vcat(func_output[i+2][np*(j-1)+1:np*j-1],                    # >>Subgradient values for the i'th variable for the j'th node
                            CUDA.zeros(Float64, w),                                         # >>Zeros for upper bound constraints (will fill in later with 1.0s)
                            0.0,                                                            # >>0.0 for the objective function row
                            0.0)                                                            # >>0.0 for the auxiliary objective function row
                            for j = 1:l]...)                                                # Repeat for every j'th node vertically
                        for i = 1:w]...,                                                  # Add a column for every i'th variable
                        [CUDA.zeros(Float64, length(b_array)) for _ = 1:2*((np-1)+w)]..., # Slack and auxiliary variables (will fill in later with 1.0s)
                        b_array)                                                          # The array of b's
        
        # Fill in the upper bound constraint indices
        for i = 1:w
            tableaus[np+i-1 : np+w+1 : end, i+2] .= 1.0 #np+w+1 length now, because of the auxiliary row
        end

        # Fill in the slack variables like normal, and then add auxiliary variables as needed
        signs = sign.(tableaus[:,end])
        signs[signs.==0] .= 1.0
        for i = 1:np+w-1
            tableaus[i:np+w+1:end, (w+2)+i] .= 1.0 #np+w+1 length now, because of the auxiliary row

            # If the "b" row is negative, do the following:
            # 1) Flip the row so that "b" is positive
            # 2) Subtract the entire row FROM the auxiliary objective row
            # 3) Add an auxiliary variable for this row
            tableaus[i:np+w+1:end, :] .*= signs[i:np+w+1:end] #Flipped the row if b was negative
            tableaus[np+w+1 : np+w+1 : end, :] .-= (signs[i:np+w+1:end].<0.0).*tableaus[i:np+w+1:end, :] #Row subtracted from auxiliary objective row
            tableaus[i:np+w+1:end, (w+2)+np+w-1+i] .+= (signs[i:np+w+1:end].<0.0).*1.0
        end

        # Send the tableaus to the parallel_simplex algorithm, with the "aux" flag set to "true"
        tableaus .= parallel_simplex(tableaus, np+w+1, aux=true)

        if all(abs.(tableaus[np+w+1:np+w+1:end,end]).<=1E-10)
            # Delete the [np+w+1 : np+w+1 : end] rows and the [w+1+(np+w-1) + 1 : end-1] columns
            # Note: is it faster to NOT remove the rows/columns and just have an adjusted simplex
            # algorithm that ignores them? Maybe, maybe not. I'll test later.
            tableaus = tableaus[setdiff(1:end, np+w+1:np+w+1:end), setdiff(1:end, w+2+(np+w-1):end-1)]
            tableaus .= parallel_simplex(tableaus, np+w)
        else
            warn = true
        end
    end

    # display(Array(func_output[2]))
    # display(Array(tableaus))
    # display(Array(-tableaus[np+w:np+w:end,end]))
    # display(Array(func_output[2][1:np:end]))
    # display(Array(max.(func_output[2][1:np:end], -tableaus[np+w:np+w:end,end])))

    # Step 8) Add results to lower and upper bound storage
    t.lower_bound_storage .= Array(max.(func_output[2][1:np:end], -tableaus[np+w:np+w:end,end]))
    t.upper_bound_storage .= Array(func_output[1][np:np:end])

    return nothing
end




# The GPU Simplex algorithm 
function parallel_simplex_older(tableau, n_rows; step_limit=100)
    # This version takes in a 2D matrix of stacked tableaus and performs the simplex
    # algorithm on the entire stacked tableau. Preallocate everything that we can.
    println("\n\nStart simplex")
    CUDA.memory_status()
    row_count = size(tableau, 1)
    n_probs = Int(row_count/n_rows)
    count_vector = CuArray{Int64}(collect(1:row_count))
    pivot_vals = CuArray{Float64}(undef, row_count)
    vals = CuArray{Int64}(undef, n_probs)
    cols = CuArray{CartesianIndex{2}}(undef, n_probs)
    col_inds = CuArray{CartesianIndex{2}}(undef, row_count)
    ratios = CuArray{Float64}(undef, row_count)
    rows = CuArray{CartesianIndex{2}}(undef, n_probs)
    pivot_rows = CuArray{Int32}(undef, row_count)
    row_inds = CuArray{CartesianIndex{2}}(undef, row_count)
    bland_set = CuArray{Float64}(undef, n_probs, size(tableau, 2)-1)
    bland_set_bool = CuArray{Bool}(undef, n_probs, size(tableau, 2)-1)
    col_ind_vec = CuArray{Float64}(undef, row_count)
    row_ind_vec = CuArray{Float64}(undef, row_count)
    pivot_tableau = CuArray{Float64}(undef, row_count, size(tableau,2))

    println("\n\nIndividual checks")
    CUDA.memory_status()

    reached = 0
    for STEP = 1:step_limit
        println("\nStep $STEP") # TOTAL: 32 MiB
        display(Array(tableau))
        reached += 1
        # Step 1: Identify the pivot columns following Bland's rule (picking
        #         the lowest variable index that has a negative value)
        bland_set .= tableau[n_rows:n_rows:end,1:end-1] # 2 MiB
        bland_set_bool .= (bland_set.<-1E-10) # None
        mins = findmax(bland_set_bool, dims=2) # 0 MiB
        vals .= -mins[1]
        cols .= mins[2]
        # @show size(vals)
        # @show size(cols)
        # @show Array(vals)
        # @show Array(cols)

        println("\nmins calculated") # 4 MiB
        CUDA.memory_status()

        if sum(vals)==0 # I.e., if we don't need to pivot on any tableau
            break
        end

        # Use the columns to generate cartesian indices we can refer to for each row
        col_inds .= CartesianIndex.(count_vector, repeat(getfield.(getfield.(cols, 1), 2), inner=(n_rows,1)))
        col_ind_vec .= tableau[col_inds]
        # @show size(col_inds)
        # @show size(col_ind_vec)
        # @show Array(col_inds)
        # @show Array(col_ind_vec)
        # println("\ncol_inds calculated") # 2 MiB
        # CUDA.memory_status()

        # Step 2: Using the identified columns, calculate ratios and pick pivot rows
        ratios .= tableau[:,end] ./ col_ind_vec
        ratios[col_ind_vec .<= 0] .= Inf
        ratios[ratios.==-Inf] .= Inf
        # @show size(ratios)
        # println("\nratios calculated") # 2 MiB
        # CUDA.memory_status()

        if all(isinf.(ratios))
            # display(Array(tableau))
            error("Pivot failed; at least one column is all negatives or 0s. Submit an issue if you get this error.")
            break
        end

        # Apply the ratio test to each pivot column separately
        # (Note: if this is an auxiliary system with auxiliary variables,
        # "aux" should be set to "true", and this step will disallow the main
        # objective function from being selected as the pivot row)
        rows .= reshape(findmin(reshape(ratios, (n_rows,:))[1:end-1,:], dims=1)[2], (:,1))
        @show rows
        @show size(rows)
        println("\nrows calculated") # None
        CUDA.memory_status()


        pivot_rows .= Int.(ceil.(count_vector./n_rows).-1).*n_rows .+ repeat(getfield.(getfield.(rows, 1), 1), inner=(n_rows, 1))
        println("\n")
        @show Array(Int.(ceil.(count_vector./n_rows).-1).*n_rows)
        @show Array(repeat(getfield.(getfield.(rows, 1), 1), inner=(n_rows, 1)))
        @show size(count_vector)
        @show Array(count_vector)
        @show n_rows
        @show size(pivot_rows)
        println("\npivot_rows calculated") # None
        CUDA.memory_status()

        row_inds .= CartesianIndex.(pivot_rows, #Rows come from the pivot rows
                                    getfield.(getfield.(col_inds, 1), 2)) #Cols are from col_inds
        row_ind_vec .= tableau[row_inds]
        println("\nrow_inds calculated") # 2 MiB
        CUDA.memory_status()

        # Step 3: Pivot! Pivot!!!
        # Find multiplication factors for each row, but set the pivot row's factor separately
        pivot_vals .= -(col_ind_vec./row_ind_vec)
        pivot_vals[getfield.(getfield.(row_inds, 1), 1)] .= (1 ./ row_ind_vec) .- 1.0
        println("\npivot_vals calculated") # 2 MiB
        CUDA.memory_status()

        # Use the original "vals" to set pivot_vals to 0 if the val was nonnegative 
        # (i.e., don't change anything if there's no need to pivot)
        pivot_vals[repeat(vals, inner=(n_rows,1)).>=0.0] .= 0.0
        println("\npivot_vals calculated") # 2 MiB
        CUDA.memory_status()
        # display(pivot_rows[:])
        # display(pivot_rows)
        # @show size(pivot_rows)
        # @show size(pivot_tableau)
        # @show size(tableau[pivot_rows[:],:])
        # error()
        # pivot_tableau .= tableau[pivot_rows[:],:]
        pivot_tableau .= tableau[pivot_rows,:]
        println("\npivot_tableau calculated") # 18 MiB
        CUDA.memory_status()

        # Adjust the tableau
        tableau .+= pivot_vals .* pivot_tableau
        println("\nmultiplication done") # None
        CUDA.memory_status()

        # Fix values to 0.0 and 1.0
        tableau[col_inds] .= 0.0
        tableau[row_inds] .= 1.0
        error()
    end

    if reached==step_limit
        # display(Array(tableau))
        error("Cycle of some sort detected; solution not guaranteed! Submit an issue if you get this error.")
    end
    for i in [count_vector, pivot_vals, vals, cols, col_inds, 
              ratios, rows, pivot_rows, row_inds, bland_set,
              col_ind_vec, row_ind_vec, pivot_tableau]
        CUDA.unsafe_free!(i)
    end

    println("\nEnd Simplex")
    CUDA.memory_status()
    return tableau #, tableau[n_rows:n_rows:end,end] would be the [negative] solutions only
end

function parallel_simplex_old(tableau, n_rows; step_limit=100)
    # This version takes in a 2D matrix of stacked tableaus and performs the simplex
    # algorithm on the entire stacked tableau. Preallocate everything that we can.
    row_count = size(tableau, 1)
    n_probs = Int(row_count/n_rows)
    count_vector = CuArray{Int64}(collect(1:row_count))
    row_adds = CuArray{Int64}(collect(0:n_rows:row_count-1))
    pivot_vals = CuArray{Float64}(undef, row_count)
    vals = CuArray{Int64}(undef, n_probs)
    cols = CuArray{CartesianIndex{2}}(undef, n_probs)
    col_inds = CuArray{CartesianIndex{2}}(undef, row_count)
    ratios = CuArray{Float64}(undef, row_count)
    rows = CuArray{CartesianIndex{2}}(undef, n_probs)
    pivot_rows = CuArray{Int32}(undef, n_probs)
    row_inds = CuArray{CartesianIndex{2}}(undef, n_probs)
    bland_set = CuArray{Float64}(undef, n_probs, size(tableau, 2)-1)
    bland_set_bool = CuArray{Bool}(undef, n_probs, size(tableau, 2)-1)
    col_ind_vec = CuArray{Float64}(undef, row_count)
    row_ind_vec = CuArray{Float64}(undef, row_count)
    pivot_tableau = CuArray{Float64}(undef, row_count, size(tableau,2))

    reached = 0
    for _ = 1:step_limit
        reached += 1
        # Step 1: Identify the pivot columns following Bland's rule (picking
        #         the lowest variable index that has a negative value)
        bland_set .= tableau[n_rows:n_rows:end,1:end-1] # 2 MiB
        bland_set_bool .= (bland_set.<-1E-10) # 
        mins = findmax(bland_set_bool, dims=2) # 
        vals .= -mins[1]
        cols .= mins[2]

        if sum(vals)==0 # I.e., if we don't need to pivot on any tableau
            break
        end

        # Use the columns to generate cartesian indices we can refer to for each row
        col_inds .= CartesianIndex.(count_vector, repeat(getfield.(getfield.(cols, 1), 2), inner=(n_rows,1)))
        col_ind_vec .= tableau[col_inds]

        # Step 2: Using the identified columns, calculate ratios and pick pivot rows
        ratios .= tableau[:,end] ./ col_ind_vec
        ratios[col_ind_vec .<= 0] .= Inf
        ratios[ratios.==-Inf] .= Inf

        if all(isinf.(ratios))
            error("Pivot failed; at least one column is all negatives or 0s. Submit an issue if you get this error.")
            break
        end

        # Apply the ratio test to each pivot column separately
        # (Note: if this is an auxiliary system with auxiliary variables,
        # "aux" should be set to "true", and this step will disallow the main
        # objective function from being selected as the pivot row)
        rows .= reshape(findmin(reshape(ratios, (n_rows,:))[1:end-1,:], dims=1)[2], (:,1))

        pivot_rows .= row_adds .+ getfield.(getfield.(rows, 1), 1)

        row_inds .= CartesianIndex.(pivot_rows,
                                    getfield.(getfield.(cols, 1), 2))


        row_ind_vec.= repeat(tableau[row_inds], inner=(n_rows,))

        # Step 3: Pivot! Pivot!!!
        # Find multiplication factors for each row, but set the pivot row's factor separately
        pivot_vals .= -(col_ind_vec./row_ind_vec)
        pivot_vals[repeat(pivot_rows, inner=(n_rows,))] .= (1 ./ row_ind_vec) .- 1.0

        # Use the original "vals" to set pivot_vals to 0 if the val was nonnegative 
        # (i.e., don't change anything if there's no need to pivot)
        pivot_vals[repeat(vals, inner=(n_rows,1)).>=0.0] .= 0.0
        pivot_tableau .= repeat(tableau[pivot_rows,:], inner=(n_rows,1))

        # Adjust the tableau
        tableau .+= pivot_vals .* pivot_tableau

        # Fix values to 0.0 and 1.0
        tableau[col_inds] .= 0.0
        tableau[row_inds] .= 1.0
    end

    if reached==step_limit
        # display(Array(tableau))
        error("Cycle of some sort detected; solution not guaranteed! Submit an issue if you get this error.")
    end
    for i in [count_vector, row_adds, pivot_vals, vals, cols, col_inds, 
              ratios, rows, pivot_rows, row_inds, bland_set, bland_set_bool,
              col_ind_vec, row_ind_vec, pivot_tableau]
        CUDA.unsafe_free!(i)
    end

    return tableau #, tableau[n_rows:n_rows:end,end] would be the [negative] solutions only
end

function parallel_simplex(tableau, n_rows; step_limit=100)
    # This version takes in a 2D matrix of stacked tableaus and performs the simplex
    # algorithm on the entire stacked tableau. Preallocate everything that we can.
    total_n_rows = Int32(size(tableau, 1))
    n_cols = Int32(size(tableau, 2))
    n_rows = Int32(n_rows) #32-bit is fine for all integers
    n_probs = Int32(total_n_rows/n_rows)
    bland_set = CuArray{Float64}(undef, n_probs, n_cols-1)
    bland_set_bool = CuArray{Bool}(undef, n_probs, n_cols-1)
    pivot_cols = CuArray{Int32}(undef, n_probs)
    pivot_col_vals = CuArray{Float64}(undef, total_n_rows)
    final_col = CuArray{Float64}(undef, total_n_rows)
    ratios = CuArray{Float64}(undef, total_n_rows)
    negatives = CuArray{Bool}(undef, total_n_rows)
    neginfs = CuArray{Bool}(undef, total_n_rows)
    ratio_bool = CuArray{Bool}(undef, total_n_rows)
    pivot_rows = CuArray{Int32}(undef, n_probs)
    blocks = Int32(cld(total_n_rows, 1024))
    
    reached = 0
    for _ = 1:step_limit
        reached += 1
        # display(Array(tableau))
        # Step 1: Identify the pivot columns following Bland's rule (picking
        #         the lowest variable index that has a negative value)
        bland_set .= @view tableau[n_rows:n_rows:end,1:end-1] # No GPU allocations
        bland_set_bool .= (bland_set.<-1E-10) # No GPU allocations

        if !(any(bland_set_bool)) # I.e., everything's 0 and we don't need to pivot anymore
            break
        end

        # Call the first-true finder kernel. This sets "pivot_cols" to the column number
        # of the first "true" in each row, or 0 if there aren't any "true"
        CUDA.@cuda blocks=n_probs threads=n_cols-1 first_true_kernel(bland_set_bool, pivot_cols)

        # Fill pivot_col_vals with the correct entries in tableau
        CUDA.@cuda blocks=blocks threads=1024 access_kernel(tableau, n_rows, pivot_cols, pivot_col_vals)
        # pivot_col_vals is the values of tableau in the pivot columns

        # Step 2: Using the identified columns, calculate ratios and pick pivot rows
        final_col .= @view tableau[:,end]
        ratios .= final_col ./ pivot_col_vals
        negatives .= (pivot_col_vals .<= 0.0)
        neginfs .= isinf.(ratios)
        
        CUDA.@cuda blocks=blocks threads=1024 set_inf_kernel(ratios, negatives)
        CUDA.@cuda blocks=blocks threads=1024 set_inf_kernel(ratios, neginfs)

        ratio_bool .= isinf.(ratios)
        # @show typeof(ratio_bool)
        # @show all(ratio_bool)
        if all(ratio_bool)
            error("Pivot failed; at least one column is all negatives or 0s. Submit an issue if you get this error.")
            break
        end

        # Apply the ratio test to each pivot column separately... essentially,
        # find the index of the minimum value of each column
        CUDA.@cuda blocks=n_probs threads=n_rows shmem=12*n_rows find_min_kernel(ratios, n_rows, pivot_rows)
        # if reached==2
        #     display(ratios)
        #     @show pivot_rows
        # end
        # "pivot_rows" is the pivot rows

        # Now that we have the pivot columns and pivot rows for each LP,
        # we can call the pivot kernel to perform the pivot. We need at least
        # as many threads as there are columns in the tableau, but ideally
        # we'd have as many threads as there are entires in the LP. And,
        # ideally, that'd be divisible by 32, but that's incredibly rare.
        # Best we can do is make sure it doesn't go above 1024.
        CUDA.@cuda blocks=n_probs threads=min(Int32(896),n_rows*n_cols) shmem=8*(n_cols+n_rows) pivot_kernel(tableau, pivot_rows, pivot_cols, n_rows, n_cols, n_rows*n_cols)
    end

    if reached==step_limit
        # display(Array(tableau))
        error("Cycle of some sort detected; solution not guaranteed! Submit an issue if you get this error.")
    end
    for i in [bland_set, bland_set_bool, pivot_cols, pivot_col_vals, final_col, 
              ratios, negatives, neginfs, ratio_bool, pivot_rows]
        CUDA.unsafe_free!(i)
    end
    CUDA.synchronize()

    return nothing #, tableau[n_rows:n_rows:end,end] would be the [negative] solutions only
end

# A 2-phase Simplex version that works with artificial variables.
function twophase_parallel_simplex(tableau, n_vars, n_rows; step_limit=100)
    # This version takes in a 2D matrix of stacked tableaus and performs the simplex
    # algorithm on the entire stacked tableau. Preallocate everything that we can.
    total_n_rows = Int32(size(tableau, 1))
    n_cols = Int32(size(tableau, 2))
    n_rows = Int32(n_rows) #32-bit is fine for all integers
    art_start = Int32(n_rows + n_vars + 1) # Since n_rows is slack_vars+2, slack+2+w+1 is the start of the artificial rows
    n_probs = Int32(total_n_rows/n_rows)
    bland_set = CuArray{Float64}(undef, n_probs, n_cols-1)
    bland_set_bool = CuArray{Bool}(undef, n_probs, n_cols-1)
    pivot_cols = CuArray{Int32}(undef, n_probs)
    pivot_col_vals = CuArray{Float64}(undef, total_n_rows)
    final_col = CuArray{Float64}(undef, total_n_rows)
    ratios = CuArray{Float64}(undef, total_n_rows)
    negatives = CuArray{Bool}(undef, total_n_rows)
    neginfs = CuArray{Bool}(undef, total_n_rows)
    ratio_bool = CuArray{Bool}(undef, total_n_rows)
    pivot_rows = CuArray{Int32}(undef, n_probs)
    blocks = Int32(cld(total_n_rows, 1024))
    flag = CuArray{Bool}(undef, 1)

    ################### PHASE 1 ###################

    # In phase I, we solve a larger system to get an initial basic feasible solution (BFS)
    # for the problem we actually want to solve. This function assumes that the input
    # system is already in a form with artificial variables that start at "art_start".

    # Solve the system normally
    reached = 0
    for _ = 1:step_limit
        reached += 1
        # Step 1: Identify the pivot columns following Bland's rule (picking
        #         the lowest variable index that has a negative value)
        bland_set .= @view tableau[n_rows:n_rows:end,1:end-1] # No GPU allocations
        bland_set_bool .= (bland_set.<-1E-10) # No GPU allocations

        # Check if we are done pivoting, by checking if every element of bland_set_bool is "false"
        flag .= true
        CUDA.@cuda blocks=blocks threads=1024 not_any_kernel(flag, bland_set_bool)
        if Array(flag)[1] # I.e., everything's 0 and we don't need to pivot anymore
            break
        end

        # Call the first-true finder kernel. This sets "pivot_cols" to the column number
        # of the first "true" in each row, or 0 if there aren't any "true"
        CUDA.@cuda blocks=n_probs threads=n_cols-1 first_true_kernel(bland_set_bool, pivot_cols)

        # Fill pivot_col_vals with the correct entries in tableau
        CUDA.@cuda blocks=blocks threads=1024 access_kernel(tableau, n_rows, pivot_cols, pivot_col_vals)
        # pivot_col_vals is the values of tableau in the pivot columns

        # Step 2: Using the identified columns, calculate ratios and pick pivot rows
        final_col .= @view tableau[:,end]
        ratios .= final_col ./ pivot_col_vals
        negatives .= (pivot_col_vals .<= 0.0)
        neginfs .= isinf.(ratios)
        ratios[n_rows-Int32(1):n_rows:end] .= Inf # Don't allow pivoting on the Phase II objective row
        # Note: No need for a special "don't pivot on the Phase I objective" rule,
        # because the pivot column value in the Phase I objective is negative,
        # by definition of it being a pivot column. The Phase I objective ratio
        # therefore gets set to Inf by the "negatives" check.
        
        CUDA.@cuda blocks=blocks threads=1024 set_inf_kernel(ratios, negatives)
        CUDA.@cuda blocks=blocks threads=1024 set_inf_kernel(ratios, neginfs)

        # Check to see if all the values in ratios are "Inf" (if so, something went wrong with the pivot)
        flag .= true
        CUDA.@cuda blocks=blocks threads=1024 all_inf_kernel(flag, ratios)
        if Array(flag)[1]
            error("Pivot failed; at least one column is all negatives or 0s. Submit an issue if you get this error.")
            break
        end

        # Apply the ratio test to each pivot column separately... essentially,
        # find the index of the minimum value of each column
        CUDA.@cuda blocks=n_probs threads=n_rows shmem=12*n_rows find_min_kernel(ratios, n_rows, pivot_rows)
        # "pivot_rows" is the pivot rows

        # Now that we have the pivot columns and pivot rows for each LP,
        # we can call the pivot kernel to perform the pivot. We need at least
        # as many threads as there are columns in the tableau, but ideally
        # we'd have as many threads as there are entires in the LP. And,
        # ideally, that'd be divisible by 32, but that's incredibly rare.
        # Best we can do is make sure it doesn't go above 1024.
        # println("RIGHT BEFORE PIVOT")
        # display(Array(tableau)[1023*(n_rows)+1 : 1025*(n_rows),:])
        CUDA.@cuda blocks=n_probs threads=min(Int32(896),n_rows*n_cols) shmem=8*(n_cols+n_rows) pivot_kernel(tableau, pivot_rows, pivot_cols, n_rows, n_cols, n_rows*n_cols)
    end

    if reached==step_limit
        # display(Array(tableau))
        error("Cycle of some sort detected; solution not guaranteed! Submit an issue if you get this error.")
    end

    ################### PHASE 2 ###################

    # In phase II, we start from the initial BFS identified from phase I, assuming
    # phase I ended with a BFS. Start by checking if a BFS could be found, and if
    # not, set the phase II objective row to be all 0's with a final lower bound
    # of +Inf (i.e., set the value to -Inf). 
    CUDA.@cuda blocks=blocks threads=1024 feasibility_check_kernel(tableau, n_rows, n_probs)

    # Remake the boolean check matrices, since we no longer care about artificial columns.
    CUDA.unsafe_free!(bland_set)
    CUDA.unsafe_free!(bland_set_bool)
    bland_set = CuArray{Float64}(undef, n_probs, art_start-1)
    bland_set_bool = CuArray{Bool}(undef, n_probs, art_start-1)

    # Now we can solve the smaller system. We can fully ignore the artificial
    # variable columns and the phase I objective row.
    reached = 0
    for _ = 1:step_limit
        # CSV.write("C:/Users/rxg20001/OneDrive - University of Connecticut/Research OneDrive/tableaus/step$STEP.csv", DataFrame(Array(tableau),:auto), delim=',')
        reached += 1
        # Step 1: Identify the pivot columns following Bland's rule (picking
        #         the lowest variable index that has a negative value). Note
        #         that we're using n_rows-1, to be the Phase II objective,
        #         and we're not including the artificial columns.
        bland_set .= @view tableau[n_rows-1:n_rows:end,1:art_start-1] # No GPU allocations
        bland_set_bool .= (bland_set.<-1E-10) # No GPU allocations
        # @show sum(Array(bland_set_bool))
        # if sum(Array(bland_set_bool))<200
        #     @show findfirst(bland_set_bool)
        # end

        # Check if we are done pivoting, by checking if every element of bland_set_bool is "false"
        flag .= true
        CUDA.@cuda blocks=blocks threads=1024 not_any_kernel(flag, bland_set_bool)
        if Array(flag)[1] # I.e., everything's 0 and we don't need to pivot anymore
            break
        end

        # Call the first-true finder kernel. This sets "pivot_cols" to the column number
        # of the first "true" in each row, or 0 if there aren't any "true"
        CUDA.@cuda blocks=n_probs threads=art_start-1 first_true_kernel(bland_set_bool, pivot_cols)

        # Fill pivot_col_vals with the correct entries in tableau
        CUDA.@cuda blocks=blocks threads=1024 access_kernel(tableau, n_rows, pivot_cols, pivot_col_vals)
        # pivot_col_vals is the values of tableau in the pivot columns

        # Step 2: Using the identified columns, calculate ratios and pick pivot rows
        final_col .= @view tableau[:,end]
        ratios .= final_col ./ pivot_col_vals
        negatives .= (pivot_col_vals .<= 0.0)
        neginfs .= isinf.(ratios)
        ratios[n_rows:n_rows:end] .= Inf # Don't allow pivoting on the Phase I objective row
        # Note: No need for a special "don't pivot on the Phase II objective" rule,
        # because the pivot column value in the Phase II objective is negative,
        # by definition of it being a pivot column. The Phase II objective ratio
        # therefore gets set to Inf by the "negatives" check.

        CUDA.@cuda blocks=blocks threads=1024 set_inf_kernel(ratios, negatives)
        CUDA.@cuda blocks=blocks threads=1024 set_inf_kernel(ratios, neginfs)

        # Check to see if all the values in ratios are "Inf" (if so, something went wrong with the pivot)
        flag .= true
        CUDA.@cuda blocks=blocks threads=1024 all_inf_kernel(flag, ratios)
        if Array(flag)[1]
            error("Pivot failed; at least one column is all negatives or 0s. Submit an issue if you get this error.")
            break
        end

        # Apply the ratio test to each pivot column separately... essentially,
        # find the index of the minimum value of each column
        CUDA.@cuda blocks=n_probs threads=n_rows shmem=12*n_rows find_min_kernel(ratios, n_rows, pivot_rows)
        # "pivot_rows" is the pivot rows

        # Now that we have the pivot columns and pivot rows for each LP,
        # we can call the pivot kernel to perform the pivot. We need at least
        # as many threads as there are columns in the tableau, but ideally
        # we'd have as many threads as there are entires in the LP. And,
        # ideally, that'd be divisible by 32, but that's incredibly rare.
        # Best we can do is make sure it doesn't go above 1024.
        CUDA.@cuda blocks=n_probs threads=min(Int32(896),n_rows*n_cols) shmem=8*(n_cols+n_rows) pivot_kernel(tableau, pivot_rows, pivot_cols, n_rows, n_cols, n_rows*n_cols)

    end

    for i in [bland_set, bland_set_bool, pivot_cols, pivot_col_vals, final_col, 
              ratios, negatives, neginfs, ratio_bool, pivot_rows]
        CUDA.unsafe_free!(i)
    end
    # CUDA.synchronize()

    return nothing #, tableau[n_rows:n_rows:end,end] would be the [negative] solutions only
end

# Utility function to pick out a node from the subvector storage and make it the "current_node".
# This is used for branching, since currently we use the EAGO branch function that uses the
# "current_node" to make branching decisions.
function make_current_node!(t::ExtendGPU, m::EAGO.GlobalOptimizer)
    prev = copy(t.node_storage[t.node_len])
    new_lower = t.lower_bound_storage[t.node_len]
    new_upper = t.upper_bound_storage[t.node_len]
    # m._upper_objective_value = copy(new_upper)
    # if m._upper_objective_value < m._global_upper_bound
    #     m._upper_solution = (prev.lower_variable_bounds .+ prev.upper_variable_bounds)./2
    # end
    t.node_len -= 1

    m._current_node = NodeBB(prev.lower_variable_bounds, prev.upper_variable_bounds,
                             prev.is_integer, prev.continuous, new_lower, new_upper,
                             prev.depth, prev.cont_depth, prev.id, prev.branch_direction,
                             prev.last_branch, prev.branch_extent)
end
make_current_node!(m::EAGO.GlobalOptimizer{R,S,Q}) where {R,S,Q<:EAGO.ExtensionType} = make_current_node!(EAGO._ext(m), m)

# (In development) A multi-start function to enable multiple runs of a solver such as IPOPT,
# before the main B&B algorithm begins
function multistart_upper!(m::EAGO.GlobalOptimizer{R,S,Q}) where {R,S,Q<:EAGO.ExtensionType}
    m._current_node = EAGO.popmin!(m._stack)
    t = EAGO._ext(m)

    if t.multistart_points > 1
        @warn "Multistart points above 1 not currently supported."
    end
    for n = 1:t.multistart_points
        upper_optimizer = EAGO._upper_optimizer(m)
        MOI.empty!(upper_optimizer)
    
        for i = 1:m._working_problem._variable_count
            m._upper_variables[i] = MOI.add_variable(upper_optimizer)
        end
        EAGO._update_upper_variables!(upper_optimizer, m)

        for i = 1:EAGO._variable_num(EAGO.FullVar(), m)
            l  = EAGO._lower_bound(EAGO.FullVar(), m, i)
            u  = EAGO._upper_bound(EAGO.FullVar(), m, i)
            v = m._upper_variables[i]
            MOI.set(upper_optimizer, MOI.VariablePrimalStart(), v, EAGO._finite_mid(l, u)) #THIS IS WHAT I WOULD CHANGE TO MAKE IT MULTI
        end

        # add constraints
        ip = m._input_problem
        EAGO._add_constraint_store_ci_linear!(upper_optimizer, ip)
        EAGO._add_constraint_store_ci_quadratic!(upper_optimizer, ip)
        #add_soc_constraints!(m, upper_optimizer)
    
        # Add nonlinear evaluation block
        MOI.set(upper_optimizer, MOI.NLPBlock(), m._working_problem._nlp_data)
        MOI.set(upper_optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(upper_optimizer, MOI.ObjectiveFunction{EAGO.SAF}(), m._working_problem._objective_saf)

        # Optimize the object
        MOI.optimize!(upper_optimizer)
        EAGO._unpack_local_nlp_solve!(m, upper_optimizer)
        EAGO.store_candidate_solution!(m)
    end
    push!(m._stack, m._current_node)
end


# Set the upper problem heuristic to only evaluate at depth 1, for now
import EAGO: default_upper_heuristic
function default_upper_heuristic(m::EAGO.GlobalOptimizer)
    bool = false
    if EAGO._current_node(m).depth==1
        bool = true
    end
    return bool
end

# Add a custom branching function that branches at the midpoint
import EAGO: select_branch_point
function select_branch_point(t::ExtendGPU, m::EAGO.GlobalOptimizer, i)
    return EAGO._mid(EAGO.BranchVar(), m, i)
end

# Disable epigraph reformation, preprocessing, and postprocessing
import EAGO: reform_epigraph_min!
function reform_epigraph_min!(m::EAGO.GlobalOptimizer)
    nothing
end
# function reform_epigraph_min!(t::ExtendGPU, m::EAGO.GlobalOptimizer)
#     nothing
# end

# reform_epigraph_min!(m::EAGO.GlobalOptimizer) = reform_epigraph_min!(_ext(m), m)



import EAGO: preprocess!
function EAGO.preprocess!(t::ExtendGPU, x::EAGO.GlobalOptimizer)
    x._preprocess_feasibility = true
end
import EAGO: postprocess!
function EAGO.postprocess!(t::ExtendGPU, x::EAGO.GlobalOptimizer)
    x._postprocess_feasibility = true
end