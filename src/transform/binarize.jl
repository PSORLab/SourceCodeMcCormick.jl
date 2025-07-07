#=
Rules for transforming narity operations into arity 1 operations
=#
function binarize!(ex::BasicSymbolic)
    exprtype(ex) in (SYM, TERM, DIV, POW) && return nothing
    if exprtype(ex)==ADD
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
    elseif exprtype(ex)==MUL
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
end
binarize!(a::Real) = error("Attempting to apply binarize!() to a Real")
