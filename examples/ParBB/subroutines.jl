
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
    # Identify the extension
    ext = EAGO._ext(m)

    # Set counts to 1
    m._iteration_count = 1
    m._node_count = 1

    # Prepare to run branch-and-bound
    EAGO.parse_global!(m)
    EAGO.presolve_global!(m)
    EAGO.print_preamble!(m)

    # Run the NLP solver to get a start-point upper bound with multi-starting
    multistart_upper!(m)

    # Fill the stack with multiple nodes for the GPU to parallelize
    prepopulate!(m)

    # Pre-allocate storage vectors
    ext.node_storage = Vector{EAGO.NodeBB}(undef, ext.node_limit)
    ext.lower_bound_storage = Vector{Float64}(undef, ext.node_limit)

    # Run branch and bound; terminate when the stack is empty or when some
    # tolerance or limit is hit
    while !EAGO.termination_check(m)
        
        # Garbage collect every 1000 iterations
        if mod(m._iteration_count, 1000)==0
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
        m._last_lower_problem_time += @elapsed lower_and_upper_problem!(m)
        EAGO.print_results!(m, true)

        for i in 1:ext.node_len
            make_current_node!(m)

            # Check for infeasibility and store the solution
            if m._lower_feasibility && !EAGO.convergence_check(m) 
                EAGO.print_results!(m, false)
                EAGO.store_candidate_solution!(m)

                # Perform post processing on each node I'm keeping track of
                m._last_postprocessing_time += @elapsed EAGO.postprocess!(m)

                # Branch the nodes if they're feasible
                if m._postprocess_feasibility
                    EAGO.branch_node!(m)
                end
            end
        end
        EAGO.set_global_lower_bound!(m)
        m._run_time = time() - m._start_time
        m._time_left = m._parameters.time_limit - m._run_time
        EAGO.log_iteration!(m)
        EAGO.print_iteration!(m)
        m._iteration_count += 1
    end

    EAGO.set_termination_status!(m)
    EAGO.set_result_status!(m)
    EAGO.print_solution!(m)
end

# Helper functions here
function prepopulate!(t::ExtendGPU, m::EAGO.GlobalOptimizer)
    if t.prepopulate == true
        println("Prepopulating with $(t.node_limit) total nodes")

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

    return nothing
end
lower_and_upper_problem!(m::EAGO.GlobalOptimizer{R,S,Q}) where {R,S,Q<:EAGO.ExtensionType} = lower_and_upper_problem!(EAGO._ext(m), m)

# Utility function to pick out a node from the subvector storage and make it the "current_node".
# This is used for branching, since currently we use the EAGO branch function that uses the
# "current_node" to make branching decisions.
function make_current_node!(t::ExtendGPU, m::EAGO.GlobalOptimizer)
    prev = copy(t.node_storage[t.node_len])
    new_lower = t.lower_bound_storage[t.node_len]
    new_upper = t.upper_bound_storage[t.node_len]
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

# Disable epigraph reformation, preprocessing, and postprocessing
import EAGO: reform_epigraph_min!
function reform_epigraph_min!(m::EAGO.GlobalOptimizer)
    nothing
end
import EAGO: preprocess!
function EAGO.preprocess!(t::ExtendGPU, x::EAGO.GlobalOptimizer)
    x._preprocess_feasibility = true
end
import EAGO: postprocess!
function EAGO.postprocess!(t::ExtendGPU, x::EAGO.GlobalOptimizer)
    x._postprocess_feasibility = true
end