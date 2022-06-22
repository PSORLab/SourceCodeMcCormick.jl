struct McCormickTransform <: AbstractTransform end
struct McCormickIntervalTransform <: AbstractTransform end

function var_names(::McCormickTransform, s::Term{Real, Base.ImmutableDict{DataType, Any}})
    arg_list = Symbol[]
    if haskey(s.metadata, ModelingToolkit.MTKParameterCtx)
        scv = genparam(Symbol(string(get_name(s))*"_cv"))
        scc = genparam(Symbol(string(get_name(s))*"_cc"))
    else
        for i in s.arguments
            push!(arg_list, get_name(i))
        end
        scv = genvar(Symbol(string(get_name(s))*"_cv"), arg_list)
        scc = genvar(Symbol(string(get_name(s))*"_cc"), arg_list)
    end
    return Symbolics.value(scv), Symbolics.value(scc)
end
function var_names(::McCormickTransform, s::Real)
    return s, s
end
function var_names(::McCormickTransform, s::Term{Real, Nothing}) #Any terms like "Differential"
    if length(s.arguments)>1
        error("Multiple arguments not supported.")
    end
    if typeof(s.arguments[1])<:Term #then it has args
        args = Symbol[]
        for i in s.arguments[1].arguments
            push!(args, get_name(i))
        end
        var = get_name(s.arguments[1])
        var_cv = genvar(Symbol(string(var)*"_cv"), args)
        var_cc = genvar(Symbol(string(var)*"_cc"), args)
    elseif typeof(s.arguments[1])<:Sym #Then it has no args
        var_cv = genparam(Symbol(string(s.arguments[1].name)*"_cv"))
        var_cc = genparam(Symbol(string(s.arguments[1].name)*"_cc"))
    else
        error("Type of argument invalid")
    end

    scv = s.f(var_cv)
    scc = s.f(var_cc)
    return Symbolics.value(scv), Symbolics.value(scc)
end
function var_names(::McCormickTransform, s::Sym) #The parameters
    scv = genparam(Symbol(string(get_name(s))*"_cv"))
    scc = genparam(Symbol(string(get_name(s))*"_cc"))
    return Symbolics.value(scv), Symbolics.value(scc)
end

function translate_initial_conditions(::McCormickTransform, prob::ODESystem, new_eqs::Vector{Equation})
    vars, params = extract_terms(new_eqs)
    var_defaults = Dict{Any, Any}()
    param_defaults = Dict{Any, Any}()

    for (key, val) in prob.defaults
        name_cv = String(get_name(key))*"_"*"cv"
        name_cc = String(get_name(key))*"_"*"cc"
        for i in vars
            if String(i.f.name)==name_cv || String(i.f.name)==name_cc
                var_defaults[i] = val
            end
        end
        for i in params
            if String(i.name)==name_cv || String(i.name)==name_cc
                param_defaults[i] = val
            end
        end
    end
    return var_defaults, param_defaults
end


function var_names(::McCormickIntervalTransform, s::Any)
    sL, sU = var_names(IntervalTransform(), s)
    scv, scc = var_names(McCormickTransform(), s)
    return sL, sU, scv, scc
end


function translate_initial_conditions(::McCormickIntervalTransform, prob::ODESystem, new_eqs::Vector{Equation})
    vars, params = extract_terms(new_eqs)
    var_defaults = Dict{Any, Any}()
    param_defaults = Dict{Any, Any}()

    for (key, val) in prob.defaults
        name_lo = String(get_name(key))*"_"*"lo"
        name_hi = String(get_name(key))*"_"*"hi"
        name_cv = String(get_name(key))*"_"*"cv"
        name_cc = String(get_name(key))*"_"*"cc"
        for i in vars
            if in(String(i.f.name), (name_lo, name_hi, name_cv, name_cc))
                var_defaults[i] = val
            end
        end
        for i in params
            if in(String(i.name), (name_lo, name_hi, name_cv, name_cc))
                param_defaults[i] = val
            end
        end
    end
    return var_defaults, param_defaults
end

# A symbolic way of evaluating the line segment between (xL, zL) and (xU, zU) at x (returns  IfElse block)
line_expr(x, xL, xU, zL, zU) = IfElse.ifelse(zU > zL, (zL*(xU - x) + zU*(x - xL))/(xU - xL), zU)

# A symbolic way of computing the mid of three numbers (returns IfElse block)
mid_expr(a, b, c) = IfElse.ifelse((a < b) && (b < c), y, IfElse.ifelse((c < b) && (b < a), b,
                    IfElse.ifelse((b < a) && (a < c), x, IfElse.ifelse((c < a) && (a < b), a, c))))

include(joinpath(@__DIR__, "rules.jl"))