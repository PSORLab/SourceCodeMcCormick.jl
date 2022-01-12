num_or_var(ex::Number) = true
num_or_var(ex::Symbol) = true
num_or_var(ex::Num) = true
num_or_var(ex::typeof(+)) = true
num_or_var(ex::typeof(*)) = true
num_or_var(ex::Any) = checkop(op(ex))
checkop(op::Symbol) = true
checkop(op::Any) = false

function num_or_var(ex::Vector)
    for i in ex
        if ~num_or_var(i)
            return false
        end
    end
    return true
end

function is_factor(ex::Expr)
    (ex.head !== :ref)
    (ex.head !== :call) && error("Only call expressions in rhs are currently supported.")
    num_or_var(ex.args[2:end])
end

# factor!(ex::Number; assignments::Vector{Assignment}) = assignments
# factor!(ex::Symbol; assignments::Vector{Assignment}) = assignments
factor!(ex::NTuple; assignments::Vector{Assignment}) = factor!(Expr(:($([i for i in ex]...))), assignments=assignments)
factor!(ex::Tuple; assignments::Vector{Assignment}) = factor!(Expr(:($([i for i in ex]...))), assignments=assignments)

function factor!(ex::Number; assignments = Assignment[])
    new = Assignment(gensym(:aux), ex)
    index = findall(x -> x.rhs==new.rhs, assignments)
    if isempty(index)
        push!(assignments, new)
    else
        p = collect(1:length(assignments))
        deleteat!(p, index[1])
        push!(p, index[1])
        assignments[:] = assignments[p]
    end
    return assignments
end
function factor!(ex::Symbol; assignments = Assignment[])
    new = Assignment(gensym(:aux), ex)
    index = findall(x -> x.rhs==new.rhs, assignments)
    if isempty(index)
        push!(assignments, new)
    else
        p = collect(1:length(assignments))
        deleteat!(p, index[1])
        push!(p, index[1])
        assignments[:] = assignments[p]
    end
    return assignments
end

function factor!(ex::Expr; assignments = Assignment[])
    binarize!(ex)
    if is_factor(ex)
        new = Assignment(gensym(:aux), ex)
        index = findall(x -> x.rhs==new.rhs, assignments)
        if isempty(index)
            push!(assignments, new)
        else
            p = collect(1:length(assignments))
            deleteat!(p, index[1])
            push!(p, index[1])
            assignments[:] = assignments[p]
        end
        return assignments
    end
    new_expr = []
    for arg in ex.args
        if num_or_var(arg)
            push!(new_expr, arg)
        else
            factor!(arg, assignments=assignments)
            push!(new_expr, assignments[end].lhs)
        end
    end
    pushfirst!(new_expr, :call)
    factor!(Expr(:($([i for i in new_expr]...))), assignments=assignments)
    return assignments
end


# Works with this example:
# x + y + z/x
# ex = Expr(:call, :+, :x, :y, Expr(:call, :/, :z, :x))
# a = factor!(ex)

# Now this works too
# x^3 + 5*y*x^2 + 3*y^2*x + 15*y^3 + 20
# ex2 = Expr(:call, :+, Expr(:call, :^, :x, 3), Expr(:call, :*, 5, :y, Expr(:call, :^, :x, 2)), Expr(:call, :*, 3, Expr(:call, :^, :y, 2), :x), Expr(:call, :*, 15, Expr(:call, :^, :y, 3)), 20)
# b = factor!(ex2)

# This example also seems to be working properly
# (x/(y/z)) + (y/z) + (z/x)
# ex3 = Expr(:call, :+, Expr(:call, :/, :x, Expr(:call, :/, :y, :z)), Expr(:call, :/, :y, :z), Expr(:call, :/, :z, :x))
# c = factor!(ex3)

# This is correct as well
# (a/(b/(c/(d/(e/(f/g))))))
# ex4 = Expr(:call, :/, :a, Expr(:call, :/, :b, Expr(:call, :/, :c, Expr(:call, :/, :d, Expr(:call, :/, :e, Expr(:call, :/, :f, :g))))))
# d = factor!(ex4)