#=
Rules for constructing interval bounding expressions
=#

# Structure used to indicate an overload with intervals is preferable
struct IntervalTransform <: AbstractTransform end

function var_names(::IntervalTransform, s::Term{Real, Base.ImmutableDict{DataType, Any}}) #The variables
    arg_list = Symbol[]
    if haskey(s.metadata, ModelingToolkit.MTKParameterCtx)
        sL = genparam(Symbol(string(get_name(s))*"_lo"))
        sU = genparam(Symbol(string(get_name(s))*"_hi"))
    else
        for i in s.arguments
            push!(arg_list, get_name(i))
        end
        sL = genvar(Symbol(string(get_name(s))*"_lo"), arg_list)
        sU = genvar(Symbol(string(get_name(s))*"_hi"), arg_list)
    end
    return Symbolics.value(sL), Symbolics.value(sU)
end
function var_names(::IntervalTransform, s::Real)
    return s, s
end
function var_names(::IntervalTransform, s::Term{Real, Nothing}) #Any terms like "Differential", or "x[1]" (NOT x[1](t))
    if typeof(s.arguments[1])<:Term #then it has typical args like "x", "y", ...
        args = Symbol[]
        for i in s.arguments[1].arguments
            push!(args, get_name(i))
        end
        var = get_name(s.arguments[1])
        var_lo = genvar(Symbol(string(var)*"_lo"), args)
        var_hi = genvar(Symbol(string(var)*"_hi"), args)
    elseif typeof(s.arguments[1])<:Sym #Then it has no typical args, i.e., x[1] has args Any[x, 1]
        if length(s.arguments)==1
            var_lo = genparam(Symbol(string(s.arguments[1].name)*"_lo"))
            var_hi = genparam(Symbol(string(s.arguments[1].name)*"_hi"))
        else
            var_lo = genparam(Symbol(string(s.arguments[1].name)*"_"*string(s.arguments[2])*"_lo"))
            var_hi = genparam(Symbol(string(s.arguments[1].name)*"_"*string(s.arguments[2])*"_hi"))
        end
    else
        error("Type of argument invalid")
    end

    sL = s.f(var_lo)
    sU = s.f(var_hi)
    return Symbolics.value(sL), Symbolics.value(sU)
end
function var_names(::IntervalTransform, s::Sym) #The parameters
    sL = genparam(Symbol(string(get_name(s))*"_lo"))
    sU = genparam(Symbol(string(get_name(s))*"_hi"))
    return Symbolics.value(sL), Symbolics.value(sU)
end

function translate_initial_conditions(::IntervalTransform, prob::ODESystem, new_eqs::Vector{Equation})
    vars, params = extract_terms(new_eqs)
    var_defaults = Dict{Any, Any}()
    param_defaults = Dict{Any, Any}()

    for (key, val) in prob.defaults
        name_lo = String(get_name(key))*"_"*"lo"
        name_hi = String(get_name(key))*"_"*"hi"
        for i in vars
            if in(String(i.f.name), (name_lo, name_hi))
                var_defaults[i] = val
            end
        end
        for i in params
            if in(String(i.name), (name_lo, name_hi))
                param_defaults[i] = val
            end
        end
    end
    return var_defaults, param_defaults
end


# Helper functions for navigating SymbolicUtils structures
get_name(x::Sym{SymbolicUtils.FnType{Tuple{Any}, Real}, Nothing}) = x.name

"""
    get_name

Take a Symbolic-type object such as `x[1,1]` and return a symbol like `:x_1_1`.
"""
function get_name(s::Term{SymbolicUtils.FnType{Tuple, Real}, Nothing})
    d = s.arguments
    new_var = string(d[1])
    for i in 2:length(d)
        new_var = new_var*"_"*string(d[i])
    end
    return Symbol(new_var)
end

function get_name(s::Term)
    if haskey(s.metadata, ModelingToolkit.MTKParameterCtx)
        d = s.arguments
        new_param = string(d[1])
        for i in 2:length(d)
            new_param = new_param*"_"*string(d[i])
        end
        return Symbol(new_param)
    else
        return get_name(s.f)
    end
end
function get_name(s::Sym)
    return s.name
end

include(joinpath(@__DIR__, "rules.jl"))