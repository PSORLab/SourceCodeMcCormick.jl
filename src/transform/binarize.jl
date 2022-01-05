#=
Rules for transform narity operations into arity 1 operations. (Works...)
=#
function binarize!(ex::Expr)
    (arity(ex) < 3) && return ex
    if op(ex) ∈ (:+, :*, :min, :max)
        new_arg = Vector{Union{Symbol, Expr}}(undef,0)
        push!(new_arg, op(ex))
        for j=3:(arity(ex) + 1) 
            push!(new_arg, ex.args[j])
        end
        ex.args[3] = binarize!(Expr(:call, new_arg...))
        resize!(ex.args, 3)
    end
    return ex
end