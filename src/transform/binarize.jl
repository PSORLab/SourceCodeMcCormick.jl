#=
Rules for transform narity operations into arity 1 operations. (Works...)
=#
function binarize!(ex::Expr)
    (arity(ex) < 3) && return ex
    if op(ex) ∈ (:+, :*, :min, :max, +, *)
        new_arg = Vector{Union{Symbol, Expr, typeof(+), typeof(*)}}(undef,0)
        push!(new_arg, op(ex))
        for j=3:(arity(ex) + 1) 
            push!(new_arg, ex.args[j])
        end
        ex.args[3] = binarize!(Expr(:call, new_arg...))
        resize!(ex.args, 3)
    end
    return ex
end

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
