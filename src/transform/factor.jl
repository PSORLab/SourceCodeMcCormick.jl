
base_term(a::Any) = false
base_term(a::Term{Real, Base.ImmutableDict{DataType,Any}}) = true
base_term(a::Term{Real, Nothing}) = (a.f==getindex)
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

factor!(ex::Num) = factor!(ex.val)
function factor!(ex::Sym{Real, Base.ImmutableDict{DataType, Any}}; eqs = Equation[])
    index = findall(x -> isequal(x.rhs,ex), eqs)
    if isempty(index)
        newsym = gensym(:aux)
        newsym = Symbol(string(newsym)[3:5] * string(newsym)[7:end])
        newvar = genvar(newsym)
        new = Equation(Symbolics.value(newvar), ex)
        push!(eqs, new)
    else
        p = collect(1:length(eqs))
        deleteat!(p, index[1])
        push!(p, index[1])
        eqs[:] = eqs[p]
    end
    return eqs
end

function factor!(ex::SymbolicUtils.Add; eqs = Equation[])
    binarize!(ex)
    if isfactor(ex)
        index = findall(x -> isequal(x.rhs,ex), eqs)
        if isempty(index)
            newsym = gensym(:aux)
            newsym = Symbol(string(newsym)[3:5] * string(newsym)[7:end])
            newvar = genvar(newsym)
            new = Equation(Symbolics.value(newvar), ex)
            push!(eqs, new)
        else
            p = collect(1:length(eqs))
            deleteat!(p, index[1])
            push!(p, index[1])
            eqs[:] = eqs[p]
        end
        return eqs
    end
    new_terms = Dict{Any, Number}()
    for (key, val) in ex.dict
        if base_term(key) && isone(val)
            new_terms[key] = val
        elseif (base_term(key))
            index = findall(x -> isequal(x.rhs,val*key), eqs)
            if isempty(index)
                newsym = gensym(:aux)
                newsym = Symbol(string(newsym)[3:5] * string(newsym)[7:end])
                newvar = genvar(newsym)
                new = Equation(Symbolics.value(newvar), val*key)
                push!(eqs, new)
                new_terms[Symbolics.value(newvar)] = 1
            else
                new_terms[eqs[index[1]].lhs] = 1
            end
        else
            factor!(val*key, eqs=eqs)
            new_terms[eqs[end].lhs] = 1
        end
    end
    new_add = SymbolicUtils.Add(Real, ex.coeff, new_terms)
    factor!(new_add, eqs=eqs)
    return eqs
end
function factor!(ex::SymbolicUtils.Mul; eqs = Equation[])
    binarize!(ex)
    if isfactor(ex)
        index = findall(x -> isequal(x.rhs,ex), eqs)
        if isempty(index)
            newsym = gensym(:aux)
            newsym = Symbol(string(newsym)[3:5] * string(newsym)[7:end])
            newvar = genvar(newsym)
            new = Equation(Symbolics.value(newvar), ex)
            push!(eqs, new)
        else
            p = collect(1:length(eqs))
            deleteat!(p, index[1])
            push!(p, index[1])
            eqs[:] = eqs[p]
        end
        return eqs
    end
    new_terms = Dict{Any, Number}()
    for (key, val) in ex.dict
        if base_term(key) && isone(val)
            new_terms[key] = val
        elseif base_term(key)
            index = findall(x -> isequal(x.rhs,key^val), eqs)
            if isempty(index)
                newsym = gensym(:aux)
                newsym = Symbol(string(newsym)[3:5] * string(newsym)[7:end])
                newvar = genvar(newsym)
                new = Equation(Symbolics.value(newvar), key^val)
                push!(eqs, new)
                new_terms[Symbolics.value(newvar)] = 1
            else
                new_terms[eqs[index[1]].lhs] = 1
            end
        else
            factor!(key^val, eqs=eqs)
            new_terms[eqs[end].lhs] = 1
        end
    end
    new_mul = SymbolicUtils.Mul(Real, ex.coeff, new_terms)
    factor!(new_mul, eqs=eqs)
    return eqs
end
function factor!(ex::SymbolicUtils.Pow; eqs = Equation[])
    if isfactor(ex)
        index = findall(x -> isequal(x.rhs,ex), eqs)
        if isempty(index)
            newsym = gensym(:aux)
            newsym = Symbol(string(newsym)[3:5] * string(newsym)[7:end])
            newvar = genvar(newsym)
            new = Equation(Symbolics.value(newvar), ex)
            push!(eqs, new)
        else
            p = collect(1:length(eqs))
            deleteat!(p, index[1])
            push!(p, index[1])
            eqs[:] = eqs[p]
        end
        return eqs
    end
    if base_term(ex.base)
        new_base = ex.base
    else
        factor!(ex.base, eqs=eqs)
        new_base = eqs[end].lhs
    end
    if base_term(ex.exp)
        new_exp = ex.exp
    else
        factor!(ex.exp, eqs=eqs)
        new_exp = eqs[end].lhs
    end
    new_pow = SymbolicUtils.Pow(new_base, new_exp)
    factor!(new_pow, eqs=eqs)
    return eqs
end
function factor!(ex::SymbolicUtils.Div; eqs = Equation[])
    if isfactor(ex)
        index = findall(x -> isequal(x.rhs,ex), eqs)
        if isempty(index)
            newsym = gensym(:aux)
            newsym = Symbol(string(newsym)[3:5] * string(newsym)[7:end])
            newvar = genvar(newsym)
            new = Equation(Symbolics.value(newvar), ex)
            push!(eqs, new)
        else
            p = collect(1:length(eqs))
            deleteat!(p, index[1])
            push!(p, index[1])
            eqs[:] = eqs[p]
        end
        return eqs
    end
    if base_term(ex.num)
        new_num = ex.num
    else
        factor!(ex.num, eqs=eqs)
        new_num = eqs[end].lhs
    end
    if base_term(ex.den)
        new_den = ex.den
    else
        factor!(ex.den, eqs=eqs)
        new_den = eqs[end].lhs
    end
    new_div = SymbolicUtils.Div(new_num, new_den)
    factor!(new_div, eqs=eqs)
    return eqs
end
function factor!(ex::Term{Real, Nothing}; eqs = Equation[])
    if isfactor(ex)
        index = findall(x -> isequal(x.rhs,ex), eqs)
        if isempty(index)
            newsym = gensym(:aux)
            newsym = Symbol(string(newsym)[3:5] * string(newsym)[7:end])
            newvar = genvar(newsym)
            new = Equation(Symbolics.value(newvar), ex)
            push!(eqs, new)
        else
            p = collect(1:length(eqs))
            deleteat!(p, index[1])
            push!(p, index[1])
            eqs[:] = eqs[p]
        end
        return eqs
    end
    new_args = []
    for arg in ex.arguments
        if base_term(arg)
            push!(new_args, arg)
        else
            factor!(arg, eqs=eqs)
            push!(new_args, eqs[end].lhs)
        end
    end
    new_func = Term(ex.f, new_args)
    factor!(new_func, eqs=eqs)
    return eqs
end
function factor!(ex::Term{Real, Base.ImmutableDict{DataType, Any}}; eqs = Equation[])
    index = findall(x -> isequal(x.rhs,ex), eqs)
    if isempty(index)
        newsym = gensym(:aux)
        newsym = Symbol(string(newsym)[3:5] * string(newsym)[7:end])
        newvar = genvar(newsym)
        new = Equation(Symbolics.value(newvar), ex)
        push!(eqs, new)
    else
        p = collect(1:length(eqs))
        deleteat!(p, index[1])
        push!(p, index[1])
        eqs[:] = eqs[p]
    end
    return eqs
end


