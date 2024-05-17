
base_term(a::Any) = false
base_term(a::Real) = true
function base_term(a::BasicSymbolic)
    exprtype(a)==SYM  && return true
    exprtype(a)==TERM && return varterm(a) || (a.f==getindex)
    return false
end

function isfactor(a::BasicSymbolic)
    if exprtype(a)==SYM
        return true
    elseif exprtype(a)==TERM
        varterm(a) || (a.f==getindex) && return true
        for i in a.arguments
            ~(base_term(i)) && return false
        end
        return true
    elseif exprtype(a)==ADD
        (~iszero(a.coeff)) && (length(a.dict)>1) && return false
        (iszero(a.coeff)) && (length(a.dict)>2) && return false
        for (key, val) in a.dict
            ~(isone(val)) && return false
            ~(base_term(key)) && return false
        end
        return true
    elseif exprtype(a)==MUL
        (~isone(a.coeff)) && (length(a.dict)>1) && return false
        (isone(a.coeff)) && (length(a.dict)>2) && return false
        for (key, val) in a.dict
            ~(isone(val)) && return false
            ~(base_term(key)) && return false
        end
        return true
    elseif exprtype(a)==DIV
        ~(base_term(a.num)) && return false
        ~(base_term(a.den)) && return false
        return true
    elseif exprtype(a)==POW
        ~(base_term(a.base)) && return false
        ~(base_term(a.exp)) && return false
        return true
    end
end

function factor!(a...)
    @warn """Use of "!" is deprecated as of v0.2.0. Please call `factor()` instead."""
    return factor(a...)
end
factor(ex::Num) = factor(ex.val)
factor(ex::Num, eqs::Vector{Equation}) = factor(ex.val, eqs=eqs)

function factor(old_ex::BasicSymbolic; eqs = Equation[])
    ex = deepcopy(old_ex)
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
    if exprtype(ex)==ADD
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
                factor(val*key, eqs=eqs)
                new_terms[eqs[end].lhs] = 1
            end
        end
        new_add = SymbolicUtils.Add(Real, ex.coeff, new_terms)
        factor(new_add, eqs=eqs)
        return eqs
    elseif exprtype(ex)==MUL
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
                factor(key^val, eqs=eqs)
                new_terms[eqs[end].lhs] = 1
            end
        end
        new_mul = SymbolicUtils.Mul(Real, ex.coeff, new_terms)
        factor(new_mul, eqs=eqs)
        return eqs
    elseif exprtype(ex)==DIV
        if base_term(ex.num)
            new_num = ex.num
        else
            factor(ex.num, eqs=eqs)
            new_num = eqs[end].lhs
        end
        if base_term(ex.den)
            new_den = ex.den
        else
            factor(ex.den, eqs=eqs)
            new_den = eqs[end].lhs
        end
        new_div = SymbolicUtils.Div(new_num, new_den)
        factor(new_div, eqs=eqs)
        return eqs
    elseif exprtype(ex)==POW
        if base_term(ex.base)
            new_base = ex.base
        else
            factor(ex.base, eqs=eqs)
            new_base = eqs[end].lhs
        end
        if base_term(ex.exp)
            new_exp = ex.exp
        else
            factor(ex.exp, eqs=eqs)
            new_exp = eqs[end].lhs
        end
        new_pow = SymbolicUtils.Pow(new_base, new_exp)
        factor(new_pow, eqs=eqs)
        return eqs
    elseif exprtype(ex)==TERM
        new_args = []
        for arg in ex.arguments
            if base_term(arg)
                push!(new_args, arg)
            else
                factor(arg, eqs=eqs)
                push!(new_args, eqs[end].lhs)
            end
        end
        new_func = SymbolicUtils.Term(ex.f, new_args)
        factor(new_func, eqs=eqs)
        return eqs
    end
    return eqs
end

