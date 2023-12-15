abstract type AbstractQuantumGate{S} end

struct QuantumGate{S <: ElementarySpace, N, M <: AbstractTensorMap{S}} <: AbstractQuantumGate{S}
	positions::NTuple{N, Int}
	op::M

	function QuantumGate{S, N, M}(pos::NTuple{N, Int}, m::M) where {S<:ElementarySpace, N, M <: AbstractTensorMap{S, N, N}}
		tmp = [pos...]
		if tmp != sort(tmp)
			pos, m = _get_norm_order(pos, m)
		end
		new{S, N, M}(pos, m)
	end
end
QuantumGate{S, N}(pos::NTuple{N, Int}, m::AbstractTensorMap) where {S<:ElementarySpace, N} = QuantumGate{S, N, typeof(m)}(pos, m)
"""
	QuantumGate(pos::NTuple{N, Int}, m::AbstractTensorMap{S}) where {S <: ElementarySpace, N}
	The index convection:
	1-body gate: 				2-body gate:
		o                         o    o
		1                         1    2

		2 						  3    4
		i                         i    i
"""
QuantumGate(pos::NTuple{N, Int}, m::AbstractTensorMap{S}) where {S <: ElementarySpace, N} = QuantumGate{S, N}(pos, m)
QuantumGate(pos::Int, m::AbstractTensorMap{S, 1, 1})  where {S <: ElementarySpace} = QuantumGate((pos,), m)


Hamiltonians.positions(g::QuantumGate) = g.positions
Hamiltonians.op(g::QuantumGate) = g.op

shift(g::QuantumGate, i::Int) = QuantumGate(positions(g) .+ i, op(g))


struct AdjointQuantumGate{S <: ElementarySpace, N, M <: AbstractTensorMap{S}} <: AbstractQuantumGate{S}
	parent::QuantumGate{S, N, M}
end

Hamiltonians.positions(g::AdjointQuantumGate) = positions(g.parent)
Hamiltonians.op(g::AdjointQuantumGate) = op(g.parent)'


shift(g::AdjointQuantumGate, i::Int) = shift(g.parent, i)'

Base.adjoint(g::QuantumGate) = AdjointQuantumGate(g)
Base.adjoint(g::AdjointQuantumGate) = g.parent



function _get_norm_order(key::NTuple{N, Int}, p) where N
	seq = sortperm([key...])
	# perm = (seq..., [s + N for s in seq]...)
	return key[seq], permute(p, Tuple(seq), Tuple(seq) .+ N)
end
