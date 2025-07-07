#=
Rules for constructing interval bounding expressions
=#

# Structure used to indicate an overload with intervals is preferable
struct IntervalTransform <: AbstractTransform end

var_names(::IntervalTransform, a::Real) = a, a
function var_names(::IntervalTransform, a::BasicSymbolic)
    if exprtype(a)==SYM
        aL = genvar(Symbol(string(get_name(a))*"_lo"))
        aU = genvar(Symbol(string(get_name(a))*"_hi"))
        return aL.val, aU.val
    elseif exprtype(a)==TERM
        if varterm(a) && typeof(a.f)<:BasicSymbolic
            arg_list = Symbol[]
            for i in a.arguments
                push!(arg_list, get_name(i))
            end
            aL = genvar(Symbol(string(get_name(a))*"_lo"), arg_list)
            aU = genvar(Symbol(string(get_name(a))*"_hi"), arg_list)
            return aL.val, aU.val
        else
            aL = genvar(Symbol(string(get_name(a))*"_lo"))
            aU = genvar(Symbol(string(get_name(a))*"_hi"))
            return aL.val, aU.val
        end
    else
        error("Reached `var_names` with an unexpected type [ADD/MUL/DIV/POW]. Check expression factorization to make sure it is being binarized correctly.")
    end
end

# function translate_initial_conditions(::IntervalTransform, prob::ODESystem, new_eqs::Vector{Equation})
#     vars, params = extract_terms(new_eqs)
#     var_defaults = Dict{Any, Any}()
#     param_defaults = Dict{Any, Any}()

#     for (key, val) in prob.defaults
#         name_lo = String(get_name(key))*"_"*"lo"
#         name_hi = String(get_name(key))*"_"*"hi"
#         for i in vars
#             if in(String(i.f.name), (name_lo, name_hi))
#                 var_defaults[i] = val
#             end
#         end
#         for i in params
#             if in(String(i.name), (name_lo, name_hi))
#                 param_defaults[i] = val
#             end
#         end
#     end
#     return var_defaults, param_defaults
# end


"""
    get_name

Take a `BasicSymbolic` object such as `x[1,1]` and return a symbol like `:xv1v1`.
Note that this supports up to 9999-indexed variables (higher than that will still
work, but the order will be wrong)
"""
function get_name(a::BasicSymbolic)
    if exprtype(a)==SYM
        return a.name
    elseif exprtype(a)==TERM
        if varterm(a)
            if a.f==getindex
                args = a.arguments
                new_var = string(args[1])
                for i in 2:lastindex(args)
                    if args[i] < 10
                        new_var = new_var * "v000" * string(args[i])
                    elseif args[i] < 100
                        new_var = new_var * "v00" * string(args[i])
                    elseif args[i] < 1000
                        new_var = new_var * "v0" * string(args[i])
                    elseif args[i] < 10000
                        new_var = new_var * "v" * string(args[i])
                    else
                        @warn "Index above 10000, order may be wrong"
                        new_var = new_var * "v" * string(args[i])
                    end
                end
                return Symbol(new_var)
            else
                return a.f.name
            end
        else
            error("Problem generating variable name. This may happen if the variable is non-standard. Please post an issue if you get this error.")
        end
    end
end
get_name(a::Num) = get_name(a.val)
get_name(a::Real) = a

include(joinpath(@__DIR__, "rules.jl"))