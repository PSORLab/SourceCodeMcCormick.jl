#=
Rules for transforming narity operations into arity 1 operations
=#
function binarize!(ex::SymbolicUtils.Add)
    (arity(ex) < 3) && return ex
    # Op is already +
    skipfirst = iszero(ex.coeff)
    newdict = Dict{Any, Number}()
    for (key, val) in ex.dict
        if skipfirst
            skipfirst = false
            continue
        end
        newdict[key] = val
        delete!(ex.dict, key)
    end
    a = SymbolicUtils.Add(Real, 0, newdict)
    binarize!(a)
    ex.dict[a] = 1
    return nothing
end
function binarize!(ex::SymbolicUtils.Mul)
    (arity(ex) < 3) && return ex
    # Op is already *
    skipfirst = isone(ex.coeff)
    newdict = Dict{Any, Number}()
    for (key, val) in ex.dict
        if skipfirst
            skipfirst = false
            continue
        end
        newdict[key] = val
        delete!(ex.dict, key)
    end
    a = SymbolicUtils.Mul(Real, 1, newdict)
    binarize!(a)
    ex.dict[a] = 1
    return nothing
end
