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
    #(isone(arity(ex)) && num_or_var(first(ex))) || num_or_var(first(ex)) && num_or_var(second(ex))
end

factor!(ex::Number; assignments::Vector{Assignment}) = nothing, assignments
factor!(ex::Symbol; assignments::Vector{Assignment}) = nothing, assignments
factor!(ex::NTuple; assignments::Vector{Assignment}) = factor!(Expr(:($([i for i in ex]...))), assignments=assignments)
factor!(ex::Tuple; assignments::Vector{Assignment}) = factor!(Expr(:($([i for i in ex]...))), assignments=assignments)

# Works with simple example, but not more complicated example (see below)
# Something to do with assignments sometimes being Vector{Assignment} and sometimes
# just being Assignment. 
function factor!(ex::Expr; assignments = Assignment[])
    if is_factor(ex)
        newsym=gensym(:aux)
        push!(assignments, Assignment(newsym, ex))
        return newsym, assignments
    end
    new_expr = []
    for arg in ex.args
        newsym, assignments = factor!(arg, assignments=assignments)
        if newsym === nothing
            push!(new_expr, arg)
        else
            push!(new_expr, newsym)
        end
    end
    pushfirst!(new_expr, :call)
    _, assignments = factor!(Expr(:($([i for i in new_expr]...))), assignments=assignments)
    return assignments
end


# Works with this example:
# x + y + z/x
ex = Expr(:call, :+, :x, :y, (:call, :/, :z, :x))
factor!(ex)

# Doesn't work with this example:
# x^3 + 5*y*x^2 + 3*y^2*x + 15*y^3 + 20
ex2 = Expr(:call, :+, (:call, :^, :x, 3), (:call, :*, 5, :y, (:call, :^, :x, 2)), (:call, :*, 3, (:call, :^, :y, 2), :x), (:call, :*, 15, (:call, :^, :y, 3)), 20)
factor!(ex2)
