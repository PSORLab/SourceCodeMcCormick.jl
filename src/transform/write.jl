
# Given a vector of equations, calculate the "levels" of a computational graph, 
# where -1 represents the base-level variables, and the highest number represents
# the original expression.
function levels(a::Vector{Equation})
    # Pull all the variables from the vector of equations
    vars = [string.(pull_vars(a)); string(a[end].lhs)]

    # Create a levels dict
    levels = Dict("" => -2.0)
    for var in vars
        levels[var] = 0.0
    end

    # Extract all variables in the LHS and RHS terms of equations
    pre_LHS = [string.(x) for x in pull_vars.(Num.(getfield.(a, :lhs)))]
    LHS = hcat(pre_LHS...)

    RHS = fill("", length(LHS), 2)
    for i in eachindex(LHS)
        RHS_vars = string.(pull_vars(Num(a[i].rhs)))
        RHS[i,1] = RHS_vars[1]
        try
            RHS[i,2] = RHS_vars[2]
        catch
        end
    end

    # For the variables, if they don't appear in the LHS, mark them as -1
    for var in vars
        if ~(var in LHS)
            levels[var] = -1.0
        end
    end

    # Loop through and repeatedly update the dictionary, based on which variables
    # appear where. 
    flag = true
    while flag
        flag = false #Continue flag. If it gets set to true, we continue
        # Scan through RHS's to make a matrix of corresponding levels
        RHS_vals = zeros(size(RHS))
        for i in eachindex(RHS)
            RHS_vals[i] = levels[RHS[i]]
        end

        # For each row, do -1's, then max
        for i in 1:size(RHS_vals, 1)
            if RHS_vals[i,1]==-1 && RHS_vals[i,2]==-2 #Function of non-aux var only
                levels[LHS[i]] = 1.0
            elseif RHS_vals[i,1]==-1 && RHS_vals[i,2]==0 #Function of aux and non-aux, but aux unknown
                flag = true
                levels[LHS[i]] = 1.0
            elseif RHS_vals[i,1]==0 && RHS_vals[i,2]==-1 #Function of aux and non-aux, but aux unknown
                flag = true
                levels[LHS[i]] = 1.0
            else #Function of aux only
                maxval = max(RHS_vals[i,1], RHS_vals[i,2])
                if levels[LHS[i]] != maxval + 1
                    flag = true
                    levels[LHS[i]] = maxval + 1
                end
            end
        end
    end
    delete!(levels, "")
    return levels
end

# Translate a vector of equations into a vector of `Graphs.Edge`s,
# which can be used to construct a SimpleDiGraph
function eqn_edges(a::Vector{Equation})
    # Create the list of edges and pull all relevant variables
    edgelist = Edge{Int}[]
    vars = [string.(get_name.(pull_vars(a))); string(a[end].lhs)]
    nums = collect(1:length(vars))
    varid = Dict()
    for i in eachindex(vars)
        varid[vars[i]] = nums[i]
    end

    
    # Extract all variables in the LHS and RHS terms of equations
    pre_LHS = [string.(get_name.(x)) for x in pull_vars.(Num.(getfield.(a, :lhs)))]
    LHS = hcat(pre_LHS...)
    LHS_id = zeros(Int, size(LHS))
    for i in eachindex(LHS_id)
        LHS_id[i] = varid[LHS[i]]
    end

    RHS_vars = pull_vars.(a)
    RHS = fill("", length(LHS), maximum(length.(RHS_vars)))
    for i in eachindex(LHS)
        for j in 1:length(RHS_vars[i])
            RHS[i,j] = string(get_name(RHS_vars[i][j]))
        end
    end
    RHS_id = zeros(Int, size(RHS))
    for i in eachindex(RHS_id)
        if RHS[i]==""
            RHS_id[i] = 0
        else
            RHS_id[i] = varid[RHS[i]]
        end
    end

    # Create edges of RHS -> LHS
    for i in eachindex(LHS_id)
        for j in eachindex(RHS_id[i,:])
            if ~iszero(RHS_id[i,j])
                push!(edgelist, Edge(RHS_id[i,j], LHS_id[i]))
            end
        end
    end
    return edgelist, vars
end

# A new topological sort that tries to minimize the number of temporary vectors
# that need to be preallocated
function topological_sort(g::SimpleDiGraph; order::Vector{Int64}=Int64[])
    for i in length(g.badjlist):-1:1 # Each i is a vector
        # Always go in order of most to least complex
        lengths = [length(g.badjlist[j]) for j in g.badjlist[i]]
        for j in g.badjlist[i][sortperm(-lengths)]
            recursive_add(g, j, order)
        end
        if ~in(i, order)
            push!(order, i)
        end
    end
    return order
end
# A recursive function. If given a graph, and a specific number to add,
# either add that number, or dig into the badjlist further.
function recursive_add(g::SimpleDiGraph, i::Int, order::Vector{Int64})
    # Given an integer, check to see if that number is already in order.
    # If so, don't do anything.
    if in(i, order)
        return nothing
    else
        # If the integer i is not in order, check to see if it depends on
        # anything
        if isempty(g.badjlist[i])
            # If there are no dependencies, add i to order
            push!(order, i)
        else
            # There are dependencies. Go through each dependency and add it.
            # Always go in order of most to least complex
            lengths = [length(g.badjlist[j]) for j in g.badjlist[i]]
            for j in g.badjlist[i][sortperm(-lengths)]
                recursive_add(g, j, order)
            end
            # Now that all the dependencies have been added, we can safely add i
            push!(order, i)
        end
    end
    return nothing
end

function combine_addition(orig_set::Vector{Equation}; maxterms::Int=8)
    # We want to collapse add's together, if the resulting expression would
    # still have fewer than 32 inputs. If no subgradients are required,
    # expressions can have up to 8 unique variables. If subgradients are
    # needed, the max allowable unique variables is:
    # Subgradient dimensions -> max terms
    # 1 -> 5
    # 2 -> 4
    # 3 -> 3
    # 4 -> 2
    # 5 -> 2
    # 6 -> 2
    set = copy(orig_set)
    LHSs = getfield.(set, :lhs)
    RHSs = pull_vars.(set)
    
    for i in eachindex(set)
        if SourceCodeMcCormick.op(set[i]) == +
            new_RHS = Num[]
            for term in RHSs[i]
                if string(term) in string.(LHSs)
                    ID = findfirst(==(string(term)), string.(LHSs))
                    if length(union(RHSs[i], RHSs[ID])) <= maxterms #Limit from CUDA
                        set[i] = substitute(set[i], term => set[ID].rhs)
                        push!(new_RHS, RHSs[ID]...)
                    else
                        push!(new_RHS, term)
                    end
                else
                    push!(new_RHS, term)
                end
            end
            RHSs[i] = new_RHS
        end
    end

    # Remove equations that are no longer needed
    rm_flag = fill(true, length(set))
    rm_flag[end] = false
    for i in 1:(length(set)-1)
        for j in eachindex(set)
            if string(LHSs[i]) in string.(RHSs[j])
                rm_flag[i] = false
            end
        end
    end
    deleteat!(set, rm_flag)

    return set
end

# Function to read and interpret a function/path name
function read_string(in_string::String)
    if length(in_string)>4 && in_string[end-2:end]==".jl"
        return in_string, in_string[1:end-3]*"_vector.jl", in_string[1:end-3]*"_kernel.jl", string(split(in_string, ('/', '\\'))[end][1:end-3])
    else
        return in_string*".jl", in_string*"_vector.jl", in_string*"_kernel.jl", in_string
    end
end

# Function to generate a unique path if one was not provided
function generate_paths()
    # Always use the current directory for simplicity
    fileID = 1
    searching = true
    while searching
        stringID = string(fileID)
        if ~isfile(joinpath(@__DIR__, "storage", "newfunc"*stringID*".jl")) && ~isfile(joinpath(@__DIR__, "storage", "newfunc"*stringID*"_vector.jl"))&& ~isfile(joinpath(@__DIR__, "storage", "newfunc"*stringID*"_kernel.jl"))
        # if ~isfile("newfunc"*stringID*".jl") && ~isfile("newfunc"*stringID*"_vector.jl")&& ~isfile("newfunc"*stringID*"_kernel.jl")
            println("Creating new Julia files:")
            # println(joinpath(@__DIR__, "newfunc"*stringID*".jl"))
            println(joinpath(@__DIR__, "storage", "newfunc"*stringID*".jl"))
            println(joinpath(@__DIR__, "storage", "newfunc"*stringID*"_vector.jl"))
            println(joinpath(@__DIR__, "storage", "newfunc"*stringID*"_kernel.jl"))
            return joinpath(@__DIR__, "storage", "newfunc"*stringID*".jl"),joinpath(@__DIR__, "storage", "newfunc"*stringID*"_vector.jl"), joinpath(@__DIR__, "storage", "newfunc"*stringID*"_kernel.jl"), "newfunc"*stringID
        end
        fileID += 1
    end
end

# Function to check that paths and function name won't cause problems
function validate_paths(path::String, path_vector::String, path_kernel::String, fname::String)
    if path[end-2:end] != ".jl"
        error("Path must end in `.jl`")
    end
    if path_vector[end-2:end] != ".jl"
        error("Path must end in `.jl`")
    end
    if path_kernel[end-2:end] != ".jl"
        error("Path must end in `.jl`")
    end
end

function factor_classifier(factors::Vector{Equation})
    # Set up the string tracker and ID vector
    strings = fill("", length(factors))
    number_vector = zeros(Int, length(factors))
    
    # Convert symbolic terms into strings
    for i in eachindex(factors)
        strings[i] = sym_to_string(factors[i].rhs)
    end

    # Identify numbers based on strings
    count = 1
    for i in eachindex(number_vector)
        if i==1
            number_vector[i] = count
            count += 1
        elseif (strings[i] != "") && (strings[i] in strings[1:(i-1)])
            number_vector[i] = number_vector[findfirst(x -> x==strings[i], strings[1:(i-1)])]
        else
            number_vector[i] = count
            count += 1
        end
    end

    return number_vector
end

sym_to_string(a::Num) = sym_to_string(a.val, "")
sym_to_string(a::BasicSymbolic) = sym_to_string(a, "")
sym_to_string(a::Num, str::String) = sym_to_string(a.val, str)
sym_to_string(a::Float64, str::String) = str .* "c"
function sym_to_string(a::BasicSymbolic, str::String)
    if exprtype(a)==SYM || (exprtype(a)==TERM && a.f==getindex)
        str *= "v"
    elseif exprtype(a)==TERM
        str *= string(a.f)
    elseif exprtype(a)==ADD && arity(a)==2
        valid_flag = true
        for pair in a.dict
            if ~isone(pair.second)
                valid_flag = false
                continue
            elseif ~(exprtype(pair.first)==SYM) && ~(exprtype(pair.first)==TERM && pair.first.f==getindex)
                valid_flag = false
                continue
            end
        end
        if valid_flag
            if ~iszero(a.coeff)
                str *= "+cv"
            else
                str *= "+vv"
            end
        end
    elseif exprtype(a)==MUL && arity(a)==2
        valid_flag = true
        for pair in a.dict
            if ~(isone(pair.second))
                valid_flag = false
                continue
            elseif ~(exprtype(pair.first)==SYM) && ~(exprtype(pair.first)==TERM && pair.first.f==getindex)
                valid_flag = false
                continue
            end
        end
        if valid_flag
            if ~isone(a.coeff)
                str *= "*cv"
            else
                str *= "*vv"
            end
        end
    elseif exprtype(a)==POW
        if typeof(a.base) <: Real
            if (exprtype(a.exp)==SYM && (exprtype(a.exp)==TERM && a.exp.f==getindex))
                str *= "^cv"
            end
        elseif (exprtype(a.base)==SYM && (exprtype(a.base)==TERM && a.base.f==getindex))
            if typeof(a.exp) <: Real
                str *= "^vc"
            elseif (exprtype(a.exp)==SYM && (exprtype(a.exp)==TERM && a.exp.f==getindex))
                str *= "^vv"
            end
        end
    elseif exprtype(a)==DIV
        if typeof(a.num) <: Real
            if (exprtype(a.den)==SYM && (exprtype(a.den)==TERM && a.den.f==getindex))
                str *= "/cv"
            end
        elseif (exprtype(a.num)==SYM && (exprtype(a.num)==TERM && a.num.f==getindex))
            if typeof(a.den) <: Real
                str *= "/vc"
            elseif (exprtype(a.den)==SYM && (exprtype(a.den)==TERM && a.den.f==getindex))
                str *= "/vv"
            end
        end
    end
    return str
end


# # This version works with more complicated structures, but then we'd need to keep
# # track of what order the variables appear in, which is overly complicated
# function sym_to_string(a::BasicSymbolic, str::String)
#     if exprtype(a)==SYM || (exprtype(a)==TERM && a.f==getindex)
#         str *= "v"
#     elseif exprtype(a)==TERM
#         str *= string(a.f)
#         str = sym_to_string(a, str)
#     elseif exprtype(a)==ADD
#         str *= string(arity(a)) * "+"
#         if ~iszero(a.coeff)
#             str *= "C"
#         end
#         for pair in a.dict
#             if ~isone(pair.second)
#                 str *= "c"
#             end
#             str = sym_to_string(pair.first, str)
#         end
#     elseif exprtype(a)==MUL
#         str *= string(arity(a)) * "*"
#         if ~isone(a.coeff)
#             str *= "C"
#         end
#         for pair in a.dict
#             str = sym_to_string(pair.first^pair.second, str)
#         end
#     elseif exprtype(a)==POW
#         str *= "^"
#         str = sym_to_string(a.base, str)
#         str = sym_to_string(a.exp, str)
#     elseif exprtype(a)==DIV
#         str *= "/"
#         str = sym_to_string(a.num, str)
#         str = sym_to_string(a.den, str)
#     end
#     str *= ","
#     return str
# end


generate_inputs(num::BasicSymbolic; constants::Vector{Num}=Num[]) = generate_inputs(Num(num), constants=constants)
function generate_inputs(num::Num; constants::Vector{Num}=Num[])
    equation = 0 ~ num
    step_1 = apply_transform(McCormickIntervalTransform(), [equation], constants=constants)
    step_2 = shrink_eqs(step_1)
    input_list = pull_vars(step_2)
    return input_list
end
generate_grad_inputs(num::BasicSymbolic, gradlist::Vector{Num}; constants::Vector{Num}=Num[]) = generate_grad_inputs(Num(num), gradlist, constants=constants)
function generate_grad_inputs(num::Num, gradlist::Vector{Num}; constants::Vector{Num}=Num[])
    cvgrad, ccgrad = grad(num, gradlist, constants=constants, expand=true)
    input_list = pull_vars(cvgrad + ccgrad)
    return input_list
end

function constant_converter(input::BasicSymbolic, constants::Vector{Num})
    if exprtype(input)==ADD && arity(input)==2 && ~iszero(input.coeff)
        new_expr = @variables(constant)[]
        new_constants = copy(constants)
        push!(new_constants, new_expr)
        for pair in input.dict
            new_expr += pair.first*pair.second
        end
        return new_expr.val, new_constants, input.coeff
    elseif exprtype(input)==MUL && arity(input)==2 && ~isone(input.coeff)
        new_expr = @variables(constant)[]
        new_constants = copy(constants)
        push!(new_constants, new_expr)
        for pair in input.dict
            new_expr *= pair.first^pair.second
        end
        return new_expr.val, new_constants, input.coeff
    elseif exprtype(input)==DIV
        if typeof(input.num)<:BasicSymbolic && typeof(input.den)<:Real
            new_expr = @variables(constant)[]
            new_constants = copy(constants)
            push!(new_constants, new_expr)
            new_expr = input.num/new_expr
            return new_expr.val, new_constants, input.den
        elseif typeof(input.num)<:Real && typeof(input.den)<:BasicSymbolic
            new_expr = @variables(constant)[]
            new_constants = copy(constants)
            push!(new_constants, new_expr)
            new_expr = new_expr/input.den
            return new_expr.val, new_constants, input.num
        else
            return input, constants, nothing
        end
    else
        return input, constants, nothing
    end
end



eval_generator(num::Num, title::String; constants::Vector{Num}=Num[]) = eval_generator(num, factor(num), title, constants)
function eval_generator(num::Num, factorized::Vector{Equation}, title::String, constants::Vector{Num})
    # To reduce the number of functions being created, we can start by classifying each factor
    # based on the math. 
    eqn_reference = factor_classifier(factorized)

    # Now we want something that'll automatically figure out what functions to make
    # and then make them. 
    funcs = []
    normal_inputs = Vector{String}[]
    for i in eachindex(factorized)
        if (i==1) || (eqn_reference[i] > maximum(eqn_reference[1:(i-1)]))
            expr, new_constants, extra_value = constant_converter(factorized[i].rhs, constants)
            f_cv = Symbol("$(title)_$(i)_cv")
            f_cc = Symbol("$(title)_$(i)_cc")
            f_lo = Symbol("$(title)_$(i)_lo")
            f_hi = Symbol("$(title)_$(i)_hi")
            out = @eval $f_cv, $f_cc, $f_lo, $f_hi, normal_order = all_evaluators(Num($(expr)), constants=$new_constants)
            normal_order = out[5]
            if ~isnothing(extra_value)
                normal_order[1] = extra_value
            end
            push!(funcs, ["$(title)_$(i)_cv", "$(title)_$(i)_cc", "$(title)_$(i)_lo", "$(title)_$(i)_hi"])
            push!(normal_inputs, string.(normal_order))
        else
            expr, new_constants, extra_value = constant_converter(factorized[i].rhs, constants)
            push!(funcs, funcs[findfirst(x -> x==eqn_reference[i], eqn_reference)])
            normal_order = generate_inputs(expr, constants=new_constants)
            if ~isnothing(extra_value)
                normal_order[1] = extra_value
            end
            push!(normal_inputs, string.(normal_order))
        end
    end
    return funcs, normal_inputs
end

grad_eval_generator(num::Num, title::String; constants::Vector{Num}=Num[]) = grad_eval_generator(num, pull_vars(num), factor(num), title, constants)
grad_eval_generator(num::Num, gradlist::Vector{Num}, title::String; constants::Vector{Num}=Num[]) = grad_eval_generator(num, gradlist, factor(num), title, constants)
function grad_eval_generator(num::Num, gradlist::Vector{Num}, factorized::Vector{Equation}, title::String, constants::Vector{Num})
    # To reduce the number of functions being created, we can start by classifying each factor
    # based on the math. 
    eqn_reference = factor_classifier(factorized)

    # Now we want something that'll automatically figure out what functions to make
    # and then make them. 
    funcs = []
    normal_inputs = Vector{String}[]
    grad_inputs = Vector{String}[]

    # Check that the length of the gradlist is <=6. Note that since each term of the subgradient
    # is independent of the other terms, if more than 6 dimensions are desired, this function can
    # simply be called with the first 6 terms, and then future terms can be substituted in as
    # needed. 
    if length(gradlist) > 6
        error("Subgradients in more than 6 dimensions currently not supported. Please submit
               an issue if you encounter this error.")
    end
    for i in eachindex(factorized)
        if (i==1) || (eqn_reference[i] > maximum(eqn_reference[1:(i-1)]))
            expr, new_constants, extra_value = constant_converter(factorized[i].rhs, constants)
            f_cv = Symbol("$(title)_$(i)_cv")
            f_cc = Symbol("$(title)_$(i)_cc")
            f_lo = Symbol("$(title)_$(i)_lo")
            f_hi = Symbol("$(title)_$(i)_hi")
            out = @eval $f_cv, $f_cc, $f_lo, $f_hi, normal_order = all_evaluators(Num($(expr)), constants=$new_constants)
            normal_order = out[5]

            df_cv = Symbol("∂$(title)_$(i)_cv")
            df_cc = Symbol("∂$(title)_$(i)_cc")
            out = @eval $df_cv, $df_cc, grad_order = all_subgradients(Num($(expr)), $gradlist, expand=true, constants=$new_constants)
            grad_order = out[3]

            if ~isnothing(extra_value)
                normal_order[1] = extra_value
                if !(isempty(grad_order)) && string(grad_order[1])=="constant" #The constant doesn't appear in the derivative for addition, but does for multiplication
                    grad_order[1] = extra_value
                end
            end
            push!(funcs, ["$(title)_$(i)_cv", "$(title)_$(i)_cc", "$(title)_$(i)_lo", "$(title)_$(i)_hi", "∂$(title)_$(i)_cv", "∂$(title)_$(i)_cc"])
            push!(normal_inputs, string.(normal_order))
            push!(grad_inputs, string.(grad_order))
        else
            expr, new_constants, extra_value = constant_converter(factorized[i].rhs, constants)
            push!(funcs, funcs[findfirst(x -> x==eqn_reference[i], eqn_reference)])
            normal_order = generate_inputs(expr, constants=new_constants)
            grad_order = generate_grad_inputs(expr, gradlist, constants=new_constants)
            if ~isnothing(extra_value)
                normal_order[1] = extra_value
                if !isempty(grad_order) && string(grad_order[1])=="constant"
                    grad_order[1] = extra_value
                end
            end
            push!(normal_inputs, string.(normal_order))
            push!(grad_inputs, string.(grad_order))
        end
    end
    return funcs, normal_inputs, grad_inputs
end


fgen(num::Num; constants::Vector{Num}=Num[], mutate::Bool=false, all_inputs::Bool=false) = fgen(num, setdiff(pull_vars(num), constants), [:all], generate_paths()..., constants, mutate, all_inputs)
fgen(num::Num, string::String; constants::Vector{Num}=Num[], mutate::Bool=false, all_inputs::Bool=false) = fgen(num, setdiff(pull_vars(num), constants), [:all], read_string(string)..., constants, mutate, all_inputs)
fgen(num::Num, outputs::Vector{Symbol}; constants::Vector{Num}=Num[], mutate::Bool=false, all_inputs::Bool=false) = fgen(num, setdiff(pull_vars(num), constants), outputs, generate_paths()..., constants, mutate, all_inputs)
fgen(num::Num, outputs::Vector{Symbol}, constants::Vector{Num}, string::String, mutate::Bool=false, all_inputs::Bool=false) = fgen(num, setdiff(pull_vars(num), constants), outputs, read_string(string)..., constants, mutate, all_inputs)
fgen(num::Num, gradlist::Vector{Num}; constants::Vector{Num}=Num[], mutate::Bool=false, all_inputs::Bool=false) = fgen(num, gradlist, [:all], generate_paths()..., constants, mutate, all_inputs)
fgen(num::Num, gradlist::Vector{Num}, outputs::Vector{Symbol}; constants::Vector{Num}=Num[], mutate::Bool=false, all_inputs::Bool=false) = fgen(num, gradlist, outputs, generate_paths()..., constants, mutate, all_inputs)
fgen(num::Num, gradlist::Vector{Num}, outputs::Vector{Symbol}, constants::Vector{Num}, mutate::Bool=false, all_inputs::Bool=false) = fgen(num, gradlist, outputs, generate_paths()..., constants, mutate, all_inputs)
fgen(num::Num, gradlist::Vector{Num}, outputs::Vector{Symbol}, string::String; constants::Vector{Num}=Num[], mutate::Bool=false, all_inputs::Bool=false) = fgen(num, gradlist, outputs, read_string(string)..., constants, mutate, all_inputs)

function fgen(num::Num, gradlist::Vector{Num}, raw_outputs::Vector{Symbol}, path::String, path_vector::String, path_kernel::String, fname::String, constants::Vector{Num}, mutate::Bool, all_inputs::Bool)
    # Ensure that paths are valid
    validate_paths(path, path_vector, path_kernel, fname)

    # Determine what objects will be returned, if mutate==false
    outputs = Symbol[]
    for output in raw_outputs
        output == :cv     && push!(outputs, :cv)
        output == :cc     && push!(outputs, :cc)
        output == :lo     && push!(outputs, :lo)
        output == :hi     && push!(outputs, :hi)
        output == :MC     && push!(outputs, :cv, :cc, :lo, :hi)
        output == :cvgrad && push!(outputs, :cvgrad)
        output == :ccgrad && push!(outputs, :ccgrad)
        output == :grad   && push!(outputs, :cvgrad, :ccgrad)
        output == :all    && push!(outputs, :cv, :cc, :lo, :hi, :cvgrad, :ccgrad)
        if ~(output in [:cv, :cc, :lo, :hi, :MC, :cvgrad, :ccgrad, :grad, :all])
            error("Output list contains an invalid output symbol: :$output. Acceptable symbols
                   include [:cv, :cc, :lo, :hi, :MC, :cvgrad, :ccgrad, :grad, :all]")
        end
    end
    if isempty(outputs)
        error("No outputs specified.")
    end
    
    # Perform a factorization of the input. If subgradients are not required, 
    # combine addition terms together in the factorization since the math is simple.
    # If subgradients are needed, we can still combine terms, but we are limited
    # further in how many terms we can combine together.
    factorized = factor(num)
    if ~(:cvgrad in outputs) && ~(:ccgrad in outputs)
        factorized = combine_addition(factorized)
    else
        factorized = combine_addition(factorized, maxterms = 5) #Would be 6-length(gradlist) for length(gradlist)<=3, but we only pass 1 grad term
    end
    
    # Generate functions from the factorization
    if (:cvgrad in outputs) || (:ccgrad in outputs)
        # Only do the first element of the gradlist, to make as few functions as possible
        funcs, normal_inputs, grad_inputs = grad_eval_generator(num, [gradlist[1]], factorized, fname, constants)
    else
        funcs, normal_inputs = eval_generator(num, factorized, fname, constants)
        grad_inputs = Vector{String}[]
    end
    # Collect LHS terms and extract original problem variables from the Num input
    LHSs = string.(getfield.(factorized, :lhs))
    if all_inputs
        vars = get_name.(gradlist)
    else
        vars = get_name.(pull_vars(num))
    end
    # vars = get_name.(pull_vars(num))
    @show vars

    # Put the factorized expression into a directed acyclic graph form
    edgelist, varids = eqn_edges(factorized) #varids includes all aux variables also
    g = SimpleDiGraph(edgelist)

    # Perform a topological sort to get the order in which we should perform
    # calculations (i.e., the final entry in "varorder" is the full original expression)
    varorder = varids[topological_sort(g)]

    # Open and begin writing information to the Julia files. Writing to files instead
    # of constructing the functions purely internally helps with debugging, and also
    # makes it easier to see how the new functions work. Note that functions cannot
    # be used or run independently of calling `fgen`, because the internal functions
    # being composed only exist at the time that the function is initially written.
    file = open(path, "w")
    file_vector = open(path_vector, "w")
    file_kernel = open(path_kernel, "w")
    files = [file, file_vector, file_kernel]
    for loc in files
        write(loc, "# Generated at $(Dates.now())\n\n")
    end

    # Determine the input list for the main function and begin writing the function
    input_list = ""
    mutate_input_list = ""
    for out in [:cv, :cc, :lo, :hi]
        if out in outputs
            mutate_input_list *= "OUT_$(string(out)), "
        end
    end
    if :cvgrad in outputs
        for dvar in string.(get_name.(gradlist))
            mutate_input_list *= "OUT_∂$(dvar)_cv, "
        end
    end
    if :ccgrad in outputs
        for dvar in string.(get_name.(gradlist))
            mutate_input_list *= "OUT_∂$(dvar)_cc, "
        end
    end

    for var in vars
        if string(var) in string.(get_name.(constants))
            input_list *= "$(var), "
            mutate_input_list *= "$(var), "
        end
    end
    for var in vars
        if ~(string(var) in string.(get_name.(constants)))
            input_list *= "$(var)_cv, $(var)_cc, $(var)_lo, $(var)_hi, "
            mutate_input_list *= "$(var)_cv, $(var)_cc, $(var)_lo, $(var)_hi, "
            if ((:cvgrad in outputs) || (:ccgrad in outputs)) && ~(string(var) in string.(get_name.(gradlist)))
                for dvar in string.(get_name.(gradlist))
                    input_list *= "∂$(var)∂$(dvar)_cv, "
                    mutate_input_list *= "∂$(var)∂$(dvar)_cv, "
                end
                for dvar in string.(get_name.(gradlist))
                    input_list *= "∂$(var)∂$(dvar)_cc, "
                    mutate_input_list *= "∂$(var)∂$(dvar)_cc, "
                end
            end
        end
    end

    input_list = input_list[1:end-2]
    mutate_input_list = mutate_input_list[1:end-2]
    write(file, "function $(fname)($(input_list)::Float64)\n")
    if mutate
        write(file_vector, "function $(fname)($(mutate_input_list)::Vector{Float64})\n")
        write(file_kernel, "function $(fname)($(mutate_input_list)::CuArray{Float64})\n")
    else
        write(file_vector, "function $(fname)($(input_list)::Vector{Float64})\n")
        write(file_kernel, "function $(fname)($(input_list)::CuArray{Float64})\n")
    end
    representative = split(input_list, " ")[end]
    println("Required inputs:")
    if mutate
        @show mutate_input_list
    else
        @show input_list
    end

    # Add in comments that describe what expression is being calculated
    for loc in files
        if mutate
            write(loc, "   # Mutate $(outputs) for the following expression: \n")
        else
            write(loc, "   # Return $(outputs) for the following expression: \n")
        end
        write(loc, "   # $(num) \n\n")
        write(loc, "   # The expression is factored into the following subexpressions,\n")
        write(loc, "   # with the functions $(fname)_i_*() referring to the i'th factor:\n")
        write(loc, "   #  FACTOR  |  EXPRESSION\n")
        L = length(string(length(factorized)))
        for i in eachindex(factorized)
            Li = length(string(i)) 
            write(loc, "   #"*" "^Int(5-floor(L/2)+(L-Li))*"$(i)"*" "^Int(5-floor((L+1)/2))*"|  $(factorized[i].lhs) = $(factorized[i].rhs)\n")
        end
    end

    # Determine how many auxiliary variables are needed
    temp_endlist = []
    maxtemp = 0
    final_only_flag = false
    for i in eachindex(varorder) # Loop through every variable that appears in the problem
        if (varorder[i] in string.(vars)) # Skip the variable if it's an input (i.e., we don't need to make an auxiliary variable for it)
            continue
        end
        if mutate && (i==length(varorder)) # If we're mutating, don't make a temporary variable for the final entry in varorder
            break
        end
        ID = findfirst(x -> occursin(varorder[i], x), varids)
        tempID = 0
        if isempty(temp_endlist)
            push!(temp_endlist, copy(g.fadjlist[ID]))
            tempID = 1
        else
            for j in eachindex(temp_endlist)
                if isempty(temp_endlist[j]) # Then we can override this one
                    temp_endlist[j] = copy(g.fadjlist[ID])
                    tempID = j
                    break
                end
            end
            if tempID==0 #Then we haven't found one we can override
                push!(temp_endlist, copy(g.fadjlist[ID]))
                tempID = length(temp_endlist)
            end
        end
        for j in eachindex(temp_endlist)
            if ID in temp_endlist[j]
                filter!(x -> x!=ID, temp_endlist[j])
            end
        end
        if tempID > maxtemp
            maxtemp = tempID
            if i==length(varorder)
                # A flag to indicate if a temporary variable is only used for the final term. This
                # allows us to only pre-allocate elements of the McCormick tuple we need to return,
                # rather than including all of [cv, cc, lo, hi, cvgrad, ccgrad] and not using some.
                final_only_flag = true 
            end
        end
    end

    # Pre-allocate space for vector and kernel versions of functions,
    # and save the names for the CUDA version to free up memory later
    write(file_vector, "   # Pre-allocate arrays for each used temp variable, similar in size to $(representative)_cv\n")
    write(file_kernel, "   # Pre-allocate CuArrays for each used temp variable, similar in size to $(representative)_cv\n")
    cuarray_list = String[]
    for loc in [file_vector, file_kernel]
        for i = 1:maxtemp-1
            write(loc, "   temp$(i)_cv = similar($(representative))\n")
            write(loc, "   temp$(i)_cc = similar($(representative))\n")
            write(loc, "   temp$(i)_lo = similar($(representative))\n")
            write(loc, "   temp$(i)_hi = similar($(representative))\n")
            if loc==file_kernel
                push!(cuarray_list, ["temp$(i)_cv", "temp$(i)_cc", "temp$(i)_lo", "temp$(i)_hi"]...)
            end
            if (:cvgrad in outputs) || (:ccgrad in outputs)
                for j in string.(get_name.(gradlist))
                    write(loc, "   ∂temp$(i)∂$(j)_cv = similar($(representative))\n")
                    if loc==file_kernel
                        push!(cuarray_list, "∂temp$(i)∂$(j)_cv")
                    end
                end
                for j in string.(get_name.(gradlist))
                    write(loc, "   ∂temp$(i)∂$(j)_cc = similar($(representative))\n")
                    if loc==file_kernel
                        push!(cuarray_list, "∂temp$(i)∂$(j)_cc")
                    end
                end
            end
        end
        if final_only_flag==false
            write(loc, "   temp$(maxtemp)_cv = similar($(representative))\n")
            write(loc, "   temp$(maxtemp)_cc = similar($(representative))\n")
            write(loc, "   temp$(maxtemp)_lo = similar($(representative))\n")
            write(loc, "   temp$(maxtemp)_hi = similar($(representative))\n")
            if loc==file_kernel
                push!(cuarray_list, ["temp$(maxtemp)_cv", "temp$(maxtemp)_cc", "temp$(maxtemp)_lo", "temp$(maxtemp)_hi"]...)
            end
            if (:cvgrad in outputs) || (:ccgrad in outputs)
                for j in string.(get_name.(gradlist))
                    write(loc, "   ∂temp$(maxtemp)∂$(j)_cv = similar($(representative))\n")
                    if loc==file_kernel
                        push!(cuarray_list, "∂temp$(maxtemp)∂$(j)_cv")
                    end
                end
                for j in string.(get_name.(gradlist))
                    write(loc, "   ∂temp$(maxtemp)∂$(j)_cc = similar($(representative))\n")
                    if loc==file_kernel
                        push!(cuarray_list, "∂temp$(maxtemp)∂$(j)_cc")
                    end
                end
            end
        else
            if :cv in outputs
                write(loc, "   temp$(maxtemp)_cv = similar($(representative))\n")
                if loc==file_kernel
                    push!(cuarray_list, "temp$(maxtemp)_cv")
                end
            end
            if :cc in outputs
                write(loc, "   temp$(maxtemp)_cc = similar($(representative))\n")
                if loc==file_kernel
                    push!(cuarray_list, "temp$(maxtemp)_cc")
                end
            end
            if :lo in outputs
                write(loc, "   temp$(maxtemp)_lo = similar($(representative))\n")
                if loc==file_kernel
                    push!(cuarray_list, "temp$(maxtemp)_lo")
                end
            end
            if :hi in outputs
                write(loc, "   temp$(maxtemp)_hi = similar($(representative))\n")
                if loc==file_kernel
                    push!(cuarray_list, "temp$(maxtemp)_hi")
                end
            end
            if :cvgrad in outputs
                for j in string.(get_name.(gradlist))
                    write(loc, "   ∂temp$(maxtemp)∂$(j)_cv = similar($(representative))\n")
                    if loc==file_kernel
                        push!(cuarray_list, "∂temp$(maxtemp)∂$(j)_cv")
                    end
                end
            end
            if :ccgrad in outputs
                for j in string.(get_name.(gradlist))
                    write(loc, "   ∂temp$(maxtemp)∂$(j)_cc = similar($(representative))\n")
                    if loc==file_kernel
                        push!(cuarray_list, "∂temp$(maxtemp)∂$(j)_cc")
                    end
                end
            end
        end
    end

    # Loop through the topological list to add calculations in order
    temp_endlist = []
    name_tracker = copy(varids)
    for i in eachindex(varorder) #Order in which variables are calculated
        # Skip calculation if the variable is one of the inputs
        if (varorder[i] in string.(vars))
            continue
        end

        # Determine the corresponding ID of the variable in varids
        ID = findfirst(x -> occursin(varorder[i], x), varids)

        # Figure out which tempID to use/override. temp_endlist keeps
        # track of where variables will be used in the future (stored
        # as g.fadjlist), with elements removed as they are used. If
        # there is an empty row in temp_endlist, we can re-use that
        # tempID. If there isn't an empty row, we add a new row.
        tempID = 0
        if isempty(temp_endlist)
            push!(temp_endlist, copy(g.fadjlist[ID]))
            tempID = 1
        else
            for j in eachindex(temp_endlist)
                if isempty(temp_endlist[j])
                    # Then we can override this one
                    temp_endlist[j] = copy(g.fadjlist[ID])
                    tempID = j
                    break
                end
            end
            if tempID==0 #Then we haven't found one we can override
                push!(temp_endlist, copy(g.fadjlist[ID]))
                tempID = length(temp_endlist)
            end
        end

        # When we refer to this variable in the future, we need to know what tempID
        # the variable is using
        name_tracker[ID] = "temp$(tempID)"

        # Prepare variable names to use for printing
        sym_cv = Symbol("$(name_tracker[ID])_cv")
        sym_cc = Symbol("$(name_tracker[ID])_cc")
        sym_lo = Symbol("$(name_tracker[ID])_lo")
        sym_hi = Symbol("$(name_tracker[ID])_hi")
        if (:cvgrad in outputs) || (:ccgrad in outputs)
            dsym_cv = Symbol[]
            dsym_cc = Symbol[]
            for j in string.(get_name.(gradlist))
                push!(dsym_cv, Symbol("∂$(name_tracker[ID])∂$(j)_cv"))
                push!(dsym_cc, Symbol("∂$(name_tracker[ID])∂$(j)_cc"))
            end
        end

        # Use inputs from the eval_generator to determine inputs for individual functions
        func_num = (findfirst(x -> occursin(varorder[i], x), LHSs))
        normal_input = ""
        for i in eachindex(normal_inputs[func_num])
            # Identify the name of the variable to be added
            name = normal_inputs[func_num][i]

            # Replace names with their tempIDs, if necessary
            for var in g.badjlist[ID]
                name = replace(name, varids[var] => name_tracker[var], count=1)
            end

            # Add to the input list
            normal_input *= name*", "
        end
        normal_input = normal_input[1:end-2]

        # Use inputs from the grad_eval_generator to determine inputs for subgradient functions.
        # Due to CUDA limitations, the max subgradient dimensionality is ~6~ 1, but we can bypass
        # this limit by calling the same functions multiple times for different sets of
        # subgradient dimensions
        if (:cvgrad in outputs) || (:ccgrad in outputs)
            grad_input = ["" for _ in 1:length(gradlist)]
            for i in eachindex(grad_inputs[func_num]) #E.g.: Num[a_cv, a_cc, a_lo, a_hi, dada_cv, dada_cc, b_cv, b_cc, b_lo, b_hi, dbda_cv, dbda_cc])
                # Identify the name of the variable to be added
                name = grad_inputs[func_num][i]

                # Replace names with their tempIDs, if necessary
                for var in g.badjlist[ID]
                    if varids[var] != name_tracker[var]
                        name = replace(name, varids[var] => name_tracker[var], count=1)
                    end
                end

                # Adjust derivative terms for the correct subgradient
                gradlen = length(gradlist)
                for j in eachindex(grad_input)
                    tempname = name
                    # @show tempname

                    # if (j>1) && (gradlen > 6)
                    #     for k in 1:6
                    #         @show k
                    #         if gradlen >= 6*j + k
                    #             println("We're inside, making this replacement:")
                    #             println("replace(tempname, ∂"*string(gradlist[k])*"_ => ∂"*string(gradlist[6*j + k])*"_")
                    #         elseif contains(tempname, "∂"*string(gradlist[k])*"_")
                    #             # If we have a dimensionality not divisible by 6, some subgradients
                    #             # will be passed as 0's
                    #             println("Otherwise, we have the condition:")
                    #             println("contains(tempname, ∂"*string(gradlist[k])*"_)")
                    #             println("tempname = 0.0")
                    #         end
                    #     end
                    # end
                    # @show gradlist
                    # @show length(gradlist)

                    if j>1
                        tempname = replace(tempname, "∂"*string(get_name(gradlist[1]))*"_" => "∂"*string(get_name(gradlist[j]))*"_")
                    end

                    # if (j>1) && (gradlen > 6)
                    #     for k in 1:6
                    #         if gradlen >= 6*j + k
                    #             tempname = replace(tempname, "∂"*string(gradlist[k])*"_" => "∂"*string(gradlist[6*j + k])*"_")
                    #         elseif contains(tempname, "∂"*string(gradlist[k])*"_")
                    #             # If we have a dimensionality not divisible by 6, some subgradients
                    #             # will be passed as 0's
                    #             tempname = "0.0"
                    #         end
                    #     end
                    # end

                    # Check if the derivative is "obvious" (e.g. dxdx = 1, and dxdy = 0, always)
                    for m in eachindex(gradlist)
                        for n in eachindex(gradlist)
                            if m==n
                                if (tempname=="∂"*string(get_name(gradlist[m]))*"∂"*string(get_name(gradlist[n]))*"_cc") || (tempname=="∂"*string(get_name(gradlist[m]))*"∂"*string(get_name(gradlist[n]))*"_cv")
                                    tempname = "1.0"
                                end
                            else
                                if (tempname=="∂"*string(get_name(gradlist[m]))*"∂"*string(get_name(gradlist[n]))*"_cc") || (tempname=="∂"*string(get_name(gradlist[m]))*"∂"*string(get_name(gradlist[n]))*"_cv")
                                    tempname = "0.0"
                                end
                            end
                        end
                    end

                    # Add the corrected name to the grad input
                    grad_input[j] *= tempname*", "
                end
            end
            # Remove the tails
            for i in eachindex(grad_input)
                grad_input[i] = grad_input[i][1:end-2]
            end
        end
        
        # Write the function calls
        if i==length(varorder) # The final set
            for loc in files
                # Set equality type to be broadcast, in parallel cases
                if loc==file # The floating-point version, which doesn't allow broadcasting
                    eq = "="
                else
                    eq = ".="
                end
                if (loc==file) || (mutate==false)
                    write(loc, "\n   # Calculate and return $(outputs) for: \n")
                    write(loc, "   # $(factorized[func_num].lhs) = ($(factorized[func_num].rhs))\n")
                    if :cv in outputs
                        write(loc, "   $(sym_cv) $(eq) $(funcs[func_num][1]).($(normal_input))\n")
                    end
                    if :cc in outputs
                        write(loc, "   $(sym_cc) $(eq) $(funcs[func_num][2]).($(normal_input))\n")
                    end
                    if :lo in outputs
                        write(loc, "   $(sym_lo) $(eq) $(funcs[func_num][3]).($(normal_input))\n")
                    end
                    if :hi in outputs
                        write(loc, "   $(sym_hi) $(eq) $(funcs[func_num][4]).($(normal_input))\n")
                    end
                    if :cvgrad in outputs
                        for j in eachindex(dsym_cv)
                            write(loc, "   $(dsym_cv[j]) $(eq) $(funcs[func_num][5])[1].($(grad_input[j]))\n")
                        end
                    end
                    if :ccgrad in outputs
                        for j in eachindex(dsym_cc)
                            write(loc, "   $(dsym_cc[j]) $(eq) $(funcs[func_num][6])[1].($(grad_input[j]))\n")
                        end
                    end
                else # It's the vector/CuArray version and mutate==true
                    write(loc, "\n   # Mutate $(outputs) for: \n")
                    write(loc, "   # $(factorized[func_num].lhs) = ($(factorized[func_num].rhs))\n")
                    if :cv in outputs
                        write(loc, "   OUT_cv $(eq) $(funcs[func_num][1]).($(normal_input))\n")
                    end
                    if :cc in outputs
                        write(loc, "   OUT_cc $(eq) $(funcs[func_num][2]).($(normal_input))\n")
                    end
                    if :lo in outputs
                        write(loc, "   OUT_lo $(eq) $(funcs[func_num][3]).($(normal_input))\n")
                    end
                    if :hi in outputs
                        write(loc, "   OUT_hi $(eq) $(funcs[func_num][4]).($(normal_input))\n")
                    end
                    if :cvgrad in outputs
                        # for j in eachindex(dsym_cv)
                        #     @show j
                        #     @show dsym_cv
                        #     @show dsym_cv[j]
                        #     @show grad_input[j]
                        # end
                        # for dvar in string.(get_name.(gradlist))
                        #     @show dvar
                        # end
                        # for dvar in string.(get_name.(gradlist))
                        for j in eachindex(dsym_cv)
                            outvar = string(get_name.(gradlist)[j])
                            write(loc, "   OUT_∂$(outvar)_cv $(eq) $(funcs[func_num][5])[1].($(grad_input[j]))\n")
                        end
                    end
                    if :ccgrad in outputs
                        for j in eachindex(dsym_cc)
                            outvar = string(get_name.(gradlist)[j])
                            write(loc, "   OUT_∂$(outvar)_cv $(eq) $(funcs[func_num][6])[1].($(grad_input[j]))\n")
                        end
                    end
                end

                # Return the desired outputs
                file_output = ""
                if :cv in outputs
                    file_output *= string(sym_cv)*", "
                    if (loc==file_kernel) && (mutate==false)
                        deleteat!(cuarray_list, findfirst(x->x==string(sym_cv), cuarray_list))
                    end
                end
                if :cc in outputs
                    file_output *= string(sym_cc)*", "
                    if (loc==file_kernel) && (mutate==false)
                        deleteat!(cuarray_list, findfirst(x->x==string(sym_cc), cuarray_list))
                    end
                end
                if :lo in outputs
                    file_output *= string(sym_lo)*", "
                    if (loc==file_kernel) && (mutate==false)
                        deleteat!(cuarray_list, findfirst(x->x==string(sym_lo), cuarray_list))
                    end
                end
                if :hi in outputs
                    file_output *= string(sym_hi)*", "
                    if (loc==file_kernel) && (mutate==false)
                        deleteat!(cuarray_list, findfirst(x->x==string(sym_hi), cuarray_list))
                    end
                end
                if :cvgrad in outputs
                    for item in dsym_cv
                        file_output *= string(item)*", "
                        if (loc==file_kernel) && (mutate==false)
                            deleteat!(cuarray_list, findfirst(x->x==string(item), cuarray_list))
                        end
                    end
                end
                if :ccgrad in outputs
                    for item in dsym_cc
                        file_output *= string(item)*", "
                        if (loc==file_kernel) && (mutate==false)
                            deleteat!(cuarray_list, findfirst(x->x==string(item), cuarray_list))
                        end
                    end
                end
                file_output = file_output[1:end-2]

                # Clear up CUDA memory
                if loc==file_kernel
                    if ~isempty(cuarray_list)
                        clear_list = "["
                        for i in cuarray_list
                            clear_list *= i*", "
                        end
                        clear_list = clear_list[1:end-2]*"]"
                        write(loc, "\n # Clear CUDA objects from memory\n")
                        write(loc, "   for i in $(clear_list)\n")
                        write(loc, "      CUDA.unsafe_free!(i)\n")
                        write(loc, "   end\n\n")
                    end
                end

                if mutate
                    write(loc, "   return nothing")
                else
                    write(loc, "   return $file_output")
                end
            end
        else
            for loc in files
                if loc==file
                    eq = "="
                else
                    eq = ".="
                end
                write(loc, "\n   # Calculate the McCormick expansion of $(factorized[func_num].lhs) = ($(factorized[func_num].rhs)) where $(factorized[func_num].lhs)=temp$(tempID)\n")
                write(loc, "   $(sym_cv) $(eq) $(funcs[func_num][1]).($(normal_input))\n")
                write(loc, "   $(sym_cc) $(eq) $(funcs[func_num][2]).($(normal_input))\n")
                write(loc, "   $(sym_lo) $(eq) $(funcs[func_num][3]).($(normal_input))\n")
                write(loc, "   $(sym_hi) $(eq) $(funcs[func_num][4]).($(normal_input))\n")
                if (:cvgrad in outputs) || (:ccgrad in outputs)
                    for j in eachindex(dsym_cv)
                        write(loc, "   $(dsym_cv[j]) $(eq) $(funcs[func_num][5])[1].($(grad_input[j]))\n")
                    end
                    for j in eachindex(dsym_cc)
                        write(loc, "   $(dsym_cc[j]) $(eq) $(funcs[func_num][6])[1].($(grad_input[j]))\n")
                    end
                end
            end
        end

        # Remove instances of ID from the templist
        for j in eachindex(temp_endlist)
            if ID in temp_endlist[j]
                filter!(x -> x!=ID, temp_endlist[j])
            end
        end
    end

    # Wrap up the files
    for loc in files
        write(loc, "\nend")
        close(loc)
    end

    # Include the new functions
    new_func = include(path)
    include(path_vector)
    include(path_kernel)
    return new_func
end