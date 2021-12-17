num_or_var(ex::Number) = true
num_or_var(ex::Symbol) = true
num_or_var(ex::Any) = false
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

factor!(ex::Number; assignments::Vector{Assignment}) = assignments
factor!(ex::Symbol; assignments::Vector{Assignment}) = assignments
factor!(ex::NTuple; assignments::Vector{Assignment}) = factor!(Expr(:($([i for i in ex]...))), assignments=assignments)
factor!(ex::Tuple; assignments::Vector{Assignment}) = factor!(Expr(:($([i for i in ex]...))), assignments=assignments)

# Seems to be working with all examples
function factor!(ex::Expr; assignments = Assignment[])
    if is_factor(ex)
        push!(assignments, Assignment(gensym(:aux), ex))
        return assignments
    end
    new_expr = []
    for arg in ex.args
        size = length(assignments)
        assignments = factor!(arg, assignments=assignments)
        if size==length(assignments) 
            push!(new_expr, arg)
        else
            push!(new_expr, assignments[end].lhs)
        end
    end
    pushfirst!(new_expr, :call)
    assignments = factor!(Expr(:($([i for i in new_expr]...))), assignments=assignments)
    return assignments
end


# Works with this example:
# x + y + z/x
# ex = Expr(:call, :+, :x, :y, (:call, :/, :z, :x))
# a = factor!(ex)

# Now this works too
# x^3 + 5*y*x^2 + 3*y^2*x + 15*y^3 + 20
# ex2 = Expr(:call, :+, (:call, :^, :x, 3), (:call, :*, 5, :y, (:call, :^, :x, 2)), (:call, :*, 3, (:call, :^, :y, 2), :x), (:call, :*, 15, (:call, :^, :y, 3)), 20)
# b = factor!(ex2)

# This example also seems to be working properly
# (x/(y/z)) + (y/z) + (z/x)
# ex3 = Expr(:call, :+, (:call, :/, :x, (:call, :/, :y, :z)), (:call, :/, :y, :z), (:call, :/, :z, :x))
# c = factor!(ex3)

# This is correct as well
# (a/(b/(c/(d/(e/(f/g))))))
# ex4 = Expr(:call, :/, :a, (:call, :/, :b, (:call, :/, :c, (:call, :/, :d, (:call, :/, :e, (:call, :/, :f, :g))))))
# d = factor!(ex4)