abstract type AbstractQuantumCircuit{S} end

struct QuantumCircuit{S <: ElementarySpace} <: AbstractQuantumCircuit{S}
	data::Vector{AbstractQuantumGate{S}}
end

QuantumCircuit{S}() where {S <: ElementarySpace} = QuantumCircuit{S}(Vector{AbstractQuantumGate{S}}())

raw_data(x::QuantumCircuit) = x.data

Base.IteratorSize(::QuantumCircuit) = Base.HasLength()
Base.IteratorEltype(::AbstractQuantumCircuit) = Base.HasEltype()
Base.eltype(::Type{QuantumCircuit{S}}) where {S<:ElementarySpace} = AbstractQuantumGate{S}

Base.getindex(x::QuantumCircuit, i::Int) = getindex(raw_data(x), i)
Base.setindex!(x::QuantumCircuit, v,  i::Int) = setindex!(raw_data(x), v, i)
Base.length(x::QuantumCircuit) = length(raw_data(x))
Base.size(x::QuantumCircuit) = size(raw_data(x))
Base.iterate(x::QuantumCircuit) = iterate(raw_data(x))
Base.iterate(x::QuantumCircuit, state) = iterate(raw_data(x), state)
Base.eltype(x::QuantumCircuit) = eltype(typeof(x))
Base.empty!(x::QuantumCircuit) = empty!(raw_data(x))

Base.isempty(x::QuantumCircuit) = isempty(raw_data(x))
Base.firstindex(x::QuantumCircuit) = firstindex(raw_data(x))
Base.lastindex(x::QuantumCircuit) = lastindex(raw_data(x))
Base.reverse(x::QuantumCircuit) = QuantumCircuit(reverse(raw_data(x)))
Base.repeat(x::QuantumCircuit, n::Int) = QuantumCircuit(repeat(raw_data(x), n))

TK.spacetype(::Type{QuantumCircuit{S}}) where {S <: ElementarySpace} = S
TK.spacetype(x::QuantumCircuit) = spacetype(typeof(x))

Base.push!(x::QuantumCircuit{S}, s::AbstractQuantumGate{S}) where {S <: ElementarySpace} = push!(raw_data(x), s)
Base.append!(x::QuantumCircuit{S}, y::QuantumCircuit{S}) where {S <: ElementarySpace} = append!(raw_data(x), raw_data(y))
Base.append!(x::QuantumCircuit{S}, y::Vector{<:AbstractQuantumGate{S}}) where {S <: ElementarySpace} = append!(raw_data(x), y)

Base.adjoint(x::QuantumCircuit{S}) where {S <: ElementarySpace} = QuantumCircuit(Vector{AbstractQuantumGate{S}}([adjoint(x[i]) for i in length(x):-1:1]))


Base.similar(x::QuantumCircuit{S}) where {S <: ElementarySpace} = QuantumCircuit{S}()
Base.copy(x::QuantumCircuit) = QuantumCircuit(copy(raw_data(x)))

Base.:*(x::QuantumCircuit{S}, y::QuantumCircuit{S}) where {S <: ElementarySpace} = QuantumCircuit{S}(vcat(raw_data(y), raw_data(x)))

