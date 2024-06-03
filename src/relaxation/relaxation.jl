struct McCormickTransform <: AbstractTransform end
struct McCormickIntervalTransform <: AbstractTransform end


var_names(::McCormickTransform, a::Real) = a, a
function var_names(::McCormickTransform, a::BasicSymbolic)
    if exprtype(a)==SYM
        acv = genvar(Symbol(string(get_name(a))*"_cv"))
        acc = genvar(Symbol(string(get_name(a))*"_cc"))
        return acv.val, acc.val
    elseif exprtype(a)==TERM
        if varterm(a) && typeof(a.f)<:BasicSymbolic
            arg_list = Symbol[]
            for i in a.arguments
                push!(arg_list, get_name(i))
            end
            acv = genvar(Symbol(string(get_name(a))*"_cv"), arg_list)
            acc = genvar(Symbol(string(get_name(a))*"_cc"), arg_list)
            return acv.val, acc.val
        else
            acv = genvar(Symbol(string(get_name(a))*"_cv"))
            acc = genvar(Symbol(string(get_name(a))*"_cc"))
            return acv.val, acc.val
        end
    else
        error("Reached `var_names` with an unexpected type [ADD/MUL/DIV/POW]. Check expression factorization to make sure it is being binarized correctly.")
    end
end

function var_names(::McCormickIntervalTransform, a::Any)
    aL, aU = var_names(IntervalTransform(), a)
    acv, acc = var_names(McCormickTransform(), a)
    return aL, aU, acv, acc
end


# function translate_initial_conditions(::McCormickTransform, prob::ODESystem, new_eqs::Vector{Equation})
#     vars, params = extract_terms(new_eqs)
#     var_defaults = Dict{Any, Any}()
#     param_defaults = Dict{Any, Any}()

#     for (key, val) in prob.defaults
#         name_cv = String(get_name(key))*"_"*"cv"
#         name_cc = String(get_name(key))*"_"*"cc"
#         for i in vars
#             if String(i.f.name)==name_cv || String(i.f.name)==name_cc
#                 var_defaults[i] = val
#             end
#         end
#         for i in params
#             if String(i.name)==name_cv || String(i.name)==name_cc
#                 param_defaults[i] = val
#             end
#         end
#     end
#     return var_defaults, param_defaults
# end

# function translate_initial_conditions(::McCormickIntervalTransform, prob::ODESystem, new_eqs::Vector{Equation})
#     vars, params = extract_terms(new_eqs)
#     var_defaults = Dict{Any, Any}()
#     param_defaults = Dict{Any, Any}()

#     for (key, val) in prob.defaults
#         name_lo = String(get_name(key))*"_"*"lo"
#         name_hi = String(get_name(key))*"_"*"hi"
#         name_cv = String(get_name(key))*"_"*"cv"
#         name_cc = String(get_name(key))*"_"*"cc"
#         for i in vars
#             if in(String(i.f.name), (name_lo, name_hi, name_cv, name_cc))
#                 var_defaults[i] = val
#             end
#         end
#         for i in params
#             if in(String(i.name), (name_lo, name_hi, name_cv, name_cc))
#                 param_defaults[i] = val
#             end
#         end
#     end
#     return var_defaults, param_defaults
# end

# A symbolic way of evaluating the line segment between (xL, zL) and (xU, zU) at x (returns  IfElse block)
line_expr(x, xL, xU, zL, zU) = IfElse.ifelse(zU > zL, (zL*(xU - x) + zU*(x - xL))/(xU - xL), zU)

# A symbolic way of computing the mid of three numbers (returns IfElse block)
mid_expr(x, y, z) = IfElse.ifelse(x >= y, IfElse.ifelse(y >= z, y, IfElse.ifelse(y == x, y, IfElse.ifelse(z >= x, x, z))),
        IfElse.ifelse(z >= y, y, IfElse.ifelse(x >= z, x, z)))

include(joinpath(@__DIR__, "rules.jl"))