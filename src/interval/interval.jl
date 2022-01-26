#=
Rules for constructing interval bounding expressions
=#

# Structure used to indicate an overload with intervals is preferable
struct IntervalTransform <: AbstractTransform end

# Creates names for interval state variables [DEPRECATED]
# function var_names(::IntervalTransform, s::Num)
#     if s.val.metadata.value[1]==:variables
#         arg_list = Symbol[]
#         for i in s.val.arguments
#             push!(arg_list, get_name(i))
#         end
#         sL = genvar(Symbol(string(s.val.f)*"_lo"), arg_list)
#         sU = genvar(Symbol(string(s.val.f)*"_hi"), arg_list)
#     elseif s.val.metadata.parent.value[1]==:parameters
#         sL = genparam(Symbol(string(s.val.name)*"_lo"))
#         sU = genparam(Symbol(string(s.val.name)*"_hi"))
#     else
#         error("Not a variable or a parameter, check type of s")
#     end
#     return sL, sU
# end

function var_names(::IntervalTransform, s::Term{Real, Base.ImmutableDict{DataType, Any}}) #The variables
    arg_list = Symbol[]
    for i in s.arguments
        push!(arg_list, get_name(i))
    end
    sL = genvar(Symbol(string(s.f)*"_lo"), arg_list)
    sU = genvar(Symbol(string(s.f)*"_hi"), arg_list)
    return sL, sU
end
function var_names(::IntervalTransform, s::Term{Real, Nothing}) #Any terms like "Differential"
    if length(s.arguments)>1
        error("Multiple arguments not supported.")
    end
    if typeof(s.arguments[1])<:Term #then it has args
        args = Symbol[]
        for i in s.arguments[1].arguments
            push!(args, get_name(i))
        end
        var = get_name(s.arguments[1])
        var_lo = genvar(Symbol(string(var)*"_lo"), args)
        var_hi = genvar(Symbol(string(var)*"_hi"), args)
    elseif typeof(s.arguments[1])<:Sym #Then it has no args
        var_lo = genparam(Symbol(string(s.arguments[1].name)*"_lo"))
        var_hi = genparam(Symbol(string(s.arguments[1].name)*"_hi"))
    else
        error("Type of argument invalid")
    end

    sL = s.f(var_lo)
    sU = s.f(var_hi)
    return sL, sU
end
function var_names(::IntervalTransform, s::Sym) #The parameters
    sL = genparam(Symbol(string(s.name)*"_lo"))
    sU = genparam(Symbol(string(s.name)*"_hi"))
    return sL, sU
end



# function var_names(::IntervalTransform, s::Number)
#     sL = s
#     sU = s
#     sL, sU
# end

# Helper functions for navigating SymbolicUtils structures
get_name(x::Sym{SymbolicUtils.FnType{Tuple{Any}, Real}, Nothing}) = x.name

"""
Takes x[1,1] returns :x_1_1
"""
function get_name(z::Term{SymbolicUtils.FnType{Tuple, Real}, Nothing})
    d = value(z).val.f.arguments
    x = string(d[1]) 
    for i in 2:length(d)
        x = x*"_"*string(i)
    end
    Symbol(x)
end

function get_name(s::Term)
    return get_name(s.f)
end
function get_name(s::Sym)
    return s.name
end

include(joinpath(@__DIR__, "rules.jl"))