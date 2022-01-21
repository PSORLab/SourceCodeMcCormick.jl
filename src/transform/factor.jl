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
function isfactor(ex::SymbolicUtils.Add)
    (~iszero(ex.coeff)) && (length(ex.dict)>1) && return false
    (iszero(ex.coeff)) && (length(ex.dict)>2) && return false
    for (key, val) in ex.dict
        ~(isone(val)) && return false
        ~(typeof(key)<:Term) && return false
    end
    return true
end
function isfactor(ex::SymbolicUtils.Mul)
    (~isone(ex.coeff)) && (length(ex.dict)>1) && return false
    (isone(ex.coeff)) && (length(ex.dict)>2) && return false
    for (key, val) in ex.dict
        ~(isone(val)) && return false
        ~(typeof(key)<:Term) && return false
    end
    return true
end
function isfactor(ex::SymbolicUtils.Div)
    ~(typeof(ex.num)<:Term) && ~(typeof(ex.num)<:Real) && return false
    ~(typeof(ex.den)<:Term) && ~(typeof(ex.num)<:Real) && return false
    return true
end
function isfactor(ex::SymbolicUtils.Pow)
    ~(typeof(ex.base)<:Term) && ~(typeof(ex.base)<:Real) && return false
    ~(typeof(ex.exp)<:Term) && ~(typeof(ex.exp)<:Real) && return false
    return true
end


# factor!(ex::Number; assignments::Vector{Assignment}) = assignments
# factor!(ex::Symbol; assignments::Vector{Assignment}) = assignments
factor!(ex::NTuple; assignments::Vector{Assignment}) = factor!(Expr(:($([i for i in ex]...))), assignments=assignments)
factor!(ex::Tuple; assignments::Vector{Assignment}) = factor!(Expr(:($([i for i in ex]...))), assignments=assignments)

function factor!(ex::Number; assignments = Assignment[])
    println("Inside this one")
    index = findall(x -> x.rhs==ex, assignments)
    if isempty(index)
        newsym = gensym(:aux)
        newsym = Symbol(string(newsym)[3:5] * string(newsym)[7:end])
        newvar = genvar(newsym)
        new = Assignment(newvar, ex)
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
    index = findall(x -> x.rhs==ex, assignments)
    if isempty(index)
        newsym = gensym(:aux)
        newsym = Symbol(string(newsym)[3:5] * string(newsym)[7:end])
        newvar = genvar(newsym)
        new = Assignment(newvar, ex)
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
        index = findall(x -> x.rhs==ex, assignments)
        if isempty(index)
            newsym = gensym(:aux)
            newsym = Symbol(string(newsym)[3:5] * string(newsym)[7:end])
            newvar = genvar(newsym)
            new = Assignment(newvar, ex)
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
function factor!(ex::SymbolicUtils.Add; assignments = Assignment[])
    binarize!(ex)
    if isfactor(ex)
        index = findall(x -> isequal(x.rhs,ex), assignments)
        if isempty(index)
            newsym = gensym(:aux)
            newsym = Symbol(string(newsym)[3:5] * string(newsym)[7:end])
            newvar = genvar(newsym)
            new = Assignment(Symbolics.value(newvar), ex)
            push!(assignments, new)
        else
            p = collect(1:length(assignments))
            deleteat!(p, index[1])
            push!(p, index[1])
            assignments[:] = assignments[p]
        end
        return assignments
    end

    new_terms = Dict{Any, Number}()
    for (key, val) in ex.dict
        if (typeof(key)<:Term) && isone(val)
            new_terms[key] = val
        elseif (typeof(key)<:Term)
            index = findall(x -> isequal(x.rhs,val*key), assignments)
            if isempty(index)
                newsym = gensym(:aux)
                newsym = Symbol(string(newsym)[3:5] * string(newsym)[7:end])
                newvar = genvar(newsym)
                new = Assignment(Symbolics.value(newvar), val*key)
                push!(assignments, new)
                new_terms[Symbolics.value(newvar)] = 1
            else
                new_terms[assignments[index[1]].lhs] = 1
            end
        else
            factor!(key, assignments=assignments)
            new_terms[assignments[end].lhs] = 1
        end
    end
    new_add = SymbolicUtils.Add(Real, ex.coeff, new_terms)
    factor!(new_add, assignments=assignments)
    return assignments
end
function factor!(ex::SymbolicUtils.Mul; assignments = Assignment[])
    binarize!(ex)
    if isfactor(ex)
        index = findall(x -> isequal(x.rhs,ex), assignments)
        if isempty(index)
            newsym = gensym(:aux)
            newsym = Symbol(string(newsym)[3:5] * string(newsym)[7:end])
            newvar = genvar(newsym)
            new = Assignment(Symbolics.value(newvar), ex)
            push!(assignments, new)
        else
            p = collect(1:length(assignments))
            deleteat!(p, index[1])
            push!(p, index[1])
            assignments[:] = assignments[p]
        end
        return assignments
    end

    new_terms = Dict{Any, Number}()
    for (key, val) in ex.dict
        if (typeof(key)<:Term) && isone(val)
            new_terms[key] = val
        elseif (typeof(key)<:Term)
            index = findall(x -> isequal(x.rhs,key^val), assignments)
            if isempty(index)
                newsym = gensym(:aux)
                newsym = Symbol(string(newsym)[3:5] * string(newsym)[7:end])
                newvar = genvar(newsym)
                new = Assignment(Symbolics.value(newvar), key^val)
                push!(assignments, new)
                new_terms[Symbolics.value(newvar)] = 1
            else
                new_terms[assignments[index[1]].lhs] = 1
            end
        else
            factor!(key, assignments=assignments)
            new_terms[assignments[end].lhs] = 1
        end
    end
    new_mul = SymbolicUtils.Mul(Real, ex.coeff, new_terms)
    factor!(new_mul, assignments=assignments)
    return assignments
end
function factor!(ex::SymbolicUtils.Pow; assignments = Assignment[])
    if isfactor(ex)
        index = findall(x -> isequal(x.rhs,ex), assignments)
        if isempty(index)
            newsym = gensym(:aux)
            newsym = Symbol(string(newsym)[3:5] * string(newsym)[7:end])
            newvar = genvar(newsym)
            new = Assignment(Symbolics.value(newvar), ex)
            push!(assignments, new)
        else
            p = collect(1:length(assignments))
            deleteat!(p, index[1])
            push!(p, index[1])
            assignments[:] = assignments[p]
        end
        return assignments
    end

    if typeof(ex.base)<:Term
        new_base = ex.base
    else
        factor!(ex.base, assignments=assignments)
        new_base = assignments[end].lhs
    end
    if typeof(ex.exp)<:Term
        new_exp = ex.exp
    else
        factor!(ex.exp, assignments=assignments)
        new_exp = assignments[end].lhs
    end
    new_pow = SymbolicUtils.Pow(new_base, new_exp)
    factor!(new_pow, assignments=assignments)
    return assignments
end
function factor!(ex::SymbolicUtils.Div; assignments = Assignment[])
    if isfactor(ex)
        index = findall(x -> isequal(x.rhs,ex), assignments)
        if isempty(index)
            newsym = gensym(:aux)
            newsym = Symbol(string(newsym)[3:5] * string(newsym)[7:end])
            newvar = genvar(newsym)
            new = Assignment(Symbolics.value(newvar), ex)
            push!(assignments, new)
        else
            p = collect(1:length(assignments))
            deleteat!(p, index[1])
            push!(p, index[1])
            assignments[:] = assignments[p]
        end
        return assignments
    end

    if typeof(ex.num)<:Term
        new_num = ex.num
    else
        factor!(ex.num, assignments=assignments)
        new_num = assignments[end].lhs
    end
    if typeof(ex.den)<:Term
        new_den = ex.den
    else
        factor!(ex.den, assignments=assignments)
        new_den = assignments[end].lhs
    end
    new_div = SymbolicUtils.Div(new_num, new_den)
    factor!(new_div, assignments=assignments)
    return assignments
end



