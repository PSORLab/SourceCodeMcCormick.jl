num_or_var(ex::Number) = true
num_or_var(ex::Symbol) = true
function is_factor(ex::Expr)
    (ex.head !== :ref)
    (ex.head !== :call) && error("Only call expressions in rhs are currently supported.")
    (isone(arity(ex)) && num_or_var(first(ex))) || num_or_var(first(ex)) && num_or_var(second(ex))
end

function factor!(ex::Expr; assigments = Assignment[])
    if is_factor(ex)
        # PROBABLY NEED TO DO SOME OTHER STUFF HERE
        push!(assigments, Assignment(gensym(:aux), ex))
        return assigments
    end
    for arg in ex.args
        factor!(arg, assigments)
    end
end