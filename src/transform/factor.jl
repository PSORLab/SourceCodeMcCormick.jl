
base_term(a::Any) = false
base_term(a::Term{Real, Base.ImmutableDict{DataType,Any}}) = true
base_term(a::Sym) = true
base_term(a::Real) = true


function isfactor(ex::SymbolicUtils.Add)
    (~iszero(ex.coeff)) && (length(ex.dict)>1) && return false
    (iszero(ex.coeff)) && (length(ex.dict)>2) && return false
    for (key, val) in ex.dict
        ~(isone(val)) && return false
        ~(base_term(key)) && return false
    end
    return true
end
function isfactor(ex::SymbolicUtils.Mul)
    (~isone(ex.coeff)) && (length(ex.dict)>1) && return false
    (isone(ex.coeff)) && (length(ex.dict)>2) && return false
    for (key, val) in ex.dict
        ~(isone(val)) && return false
        ~(base_term(key)) && return false
    end
    return true
end
function isfactor(ex::SymbolicUtils.Div)
    ~(base_term(ex.num)) && return false
    ~(base_term(ex.den)) && return false
    return true
end
function isfactor(ex::SymbolicUtils.Pow)
    ~(base_term(ex.base)) && return false
    ~(base_term(ex.exp)) && return false
    return true
end
function isfactor(ex::Term{Real,Nothing})
    for i in ex.arguments
        ~(base_term(i)) && return false
    end
    return true
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
        if base_term(key) && isone(val)
            new_terms[key] = val
        elseif (base_term(key))
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
            factor!(val*key, assignments=assignments)
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
        if base_term(key) && isone(val)
            new_terms[key] = val
        elseif base_term(key)
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
            factor!(key^val, assignments=assignments)
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
    if base_term(ex.base)
        new_base = ex.base
    else
        factor!(ex.base, assignments=assignments)
        new_base = assignments[end].lhs
    end
    if base_term(ex.exp)
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
    if base_term(ex.num)
        new_num = ex.num
    else
        factor!(ex.num, assignments=assignments)
        new_num = assignments[end].lhs
    end
    if base_term(ex.den)
        new_den = ex.den
    else
        factor!(ex.den, assignments=assignments)
        new_den = assignments[end].lhs
    end
    new_div = SymbolicUtils.Div(new_num, new_den)
    factor!(new_div, assignments=assignments)
    return assignments
end
function factor!(ex::Term{Real, Nothing}; assignments = Assignment[])
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
    new_args = []
    for arg in ex.arguments
        if base_term(arg)
            push!(new_args, arg)
        else
            factor!(arg, assignments=assignments)
            push!(new_args, assignments[end].lhs)
        end
    end
    new_func = Term(ex.f, new_args)
    factor!(new_func, assignments=assignments)
    return assignments
end
function factor!(ex::Term{Real, Base.ImmutableDict{DataType, Any}}; assignments = Assignment[])
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


