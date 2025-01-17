

function _swap_gate(svectorj1, mpsj1, svectorj2, mpsj2, trunc::TruncationScheme)
	svectorj1 = convert(TensorMap, svectorj1)
	@tensor twositemps[-1 -3; -2 -4] := mpsj1[-1, -2, 1] * mpsj2[1, -3, -4]
	@tensor twositemps1[-1 -2; -3 -4] := svectorj1[-1, 1] * twositemps[1, -2, -3, -4]
	u, s, v, err = stable_tsvd!(twositemps1, trunc=trunc)
	@tensor u[-1 -2; -3] = twositemps[-1,-2,1,2] * conj(v[-3,1,2])
	return u, s, permute(v, (1,2), (3,)), err
end

function bond_evolution(bondmpo, svectorj1, mpsj1, svectorj2, mpsj2, trunc::TruncationScheme)
	svectorj1 = convert(TensorMap, svectorj1)
	@tensor twositemps[-1 -2; -3 -4] :=  mpsj1[-1, 1, 3] * mpsj2[3, 2, -4] * bondmpo[-2,-3, 1,2]
	@tensor twositemps1[-1 -2; -3 -4] := svectorj1[-1, 1] * twositemps[1, -2, -3, -4]
	# to remove very small numbers
	u, s, v, err = stable_tsvd!(twositemps1, trunc=trunc)
	@tensor u[-1 -2; -3] = twositemps[-1,-2,1,2] * conj(v[-3,1,2])
	return u, s, permute(v, (1,2), (3,)), err	
end


function bond_evolution3(bondmpo, svectorj1, mpsj1, svectorj2, mpsj2, svectorj3, mpsj3, trunc::TruncationScheme)
	svectorj1 = convert(TensorMap, svectorj1)
	@tensor threesitemps[-1 -2; -3 -4 -5] := mpsj1[-1,3,1] * mpsj2[1,4,2] * mpsj3[2,5,-5] * bondmpo[-2,-3,-4, 3,4,5]
	@tensor threesitemps1[-1 -2; -3 -4 -5] := svectorj1[-1, 1] * threesitemps[1,-2,-3,-4,-5]
	# threesitemps1.purge()
	u, s, v, err1 = stable_tsvd!(threesitemps1, trunc=trunc)
	@tensor u[-1 -2; -3] = threesitemps[-1,-2,1,2,3] * conj(v[-3,1,2,3])
	s′ = convert(TensorMap, s)
	@tensor v1[-1 -2; -3 -4] := s′[-1, 1] * v[1,-2,-3,-4]

	u2, s2, v2, err2 = stable_tsvd!(v1, trunc=trunc)
	@tensor u2[-1 -2; -3] = v[-1,-2,1,2] *conj(v2[-3,1,2])
	return u, s, u2, s2, permute(v2, (1,2), (3,)), max(err1, err2)
end

function bond_evolution4(bondmpo, svectorj1, mpsj1, svectorj2, mpsj2, svectorj3, mpsj3, svectorj4, mpsj4, trunc::TruncationScheme)
	@tensor foursitemps[-1 -2; -3 -4 -5 -6] := mpsj1[-1,4,1] * mpsj2[1,5,2] * mpsj3[2,6,3] * mpsj4[3,7,-6] * bondmpo[-2,-3,-4,-5, 4,5,6,7]
	@tensor foursitemps1[-1 -2; -3 -4 -5 -6] := svectorj1[-1, 1] * foursitemps[1,-2,-3,-4,-5,-6]

	u, s, v, err1 = stable_tsvd!(foursitemps1, trunc=trunc)
	@tensor u[-1 -2; -3] = foursitemps[-1,-2,1,2,3,4] * conj(v[-3,1,2,3,4])
	s′ = convert(TensorMap, s)
	@tensor v1[-1 -2; -3 -4 -5] := s′[-1, 1] * v[1, -2,-3,-4,-5]
	# v1.purge()
	u2, s2, v2, err2 = stable_tsvd!(v1, trunc=trunc)
	@tensor u2[-1 -2; -3] = v[-1, -2, 1,2,3] * conj(v2[-3, 1,2,3])
	v = v2
	s2′ = convert(TensorMap, s2)
	@tensor v1[-1 -2; -3 -4] := s2′[-1, 1] * v[1,-2,-3,-4]
	u3, s3, v3, err3 = stable_tsvd!(v1, trunc=trunc)
	@tensor u3[-1 -2; -3] = v[-1,-2,1,2] * conj(v3[-3,1,2])
	return u, s, u2, s2, u3, s3, permute(v3, (1,2), (3,)), max(err1, err2, err3)
end

function _move!(psi, i::Int, j::Int, trunc::TruncationScheme)
	L = length(psi)
	(i <= L && j <= L) || throw(BoundsError("index out of range."))
	svectors_uninitialized(psi) && canonicalize!(psi)
	err = 0.
	(i==j) && return err

	err2 = 0.
	if i < j
		for k = i:(j-1)
			psi[k], psi.s[k+1], psi[k+1], err2 = _swap_gate(psi.s[k], psi[k], psi.s[k+1], psi[k+1], trunc)
			err = max(err, err2)
		end
	else
		for k = i:-1:(j+1)
			psi[k-1], psi.s[k], psi[k], err2 = _swap_gate(psi.s[k-1], psi[k-1], psi.s[k], psi[k], trunc)
			err = max(err, err2)
		end
	end
	return err
end


function _apply_impl(kk::Tuple{Int}, m::AbstractTensorMap{<:Number, S, 1, 1}, state, trunc::TruncationScheme) where S
	key = kk[1]
	@tensor tmp[-1 -2; -3] := m[-2, 1] * state[key][-1,1,-3]
	state[key] = tmp
	return 0.
end

function _apply_impl(key::Tuple{Int, Int}, m::AbstractTensorMap{<:Number, S, 2, 2}, mps, trunc::TruncationScheme) where S
	i, j = key
	if i+1==j
		mps[j-1], mps.s[j], mps[j], err = bond_evolution(m, mps.s[j-1], mps[j-1], mps.s[j], mps[j], trunc)
		return err
	end
	# forward swap
	err = 0.
	err2 = _move!(mps, i, j-1, trunc)
	err = max(err, err2)
	mps[j-1], mps.s[j], mps[j], err2 = bond_evolution(m, mps.s[j-1], mps[j-1], mps.s[j], mps[j], trunc)
	err = max(err, err2)
	# backward swap
	err2 = _move!(mps, j-1, i, trunc)
	err = max(err, err2)
	return err
end

is_nn_pos(key::Tuple{Int}) = error("input should have at least two elements.")
function is_nn_pos(key::NTuple{N, Int}) where N
	a = key[1]
	for i in 2:N
		(key[i] == a + 1) || return false
		a = key[i]
	end
	return true
end

function _apply_impl(key::Tuple{Int, Int, Int}, m::AbstractTensorMap{<:Number, S, 3, 3}, mps, trunc::TruncationScheme) where S
	i, j, k = key
	if is_nn_pos(key)
		mps[j-1], mps.s[j], mps[j], mps.s[j+1], mps[j+1], err = bond_evolution3(
			m, mps.s[j-1], mps[j-1], mps.s[j], mps[j], mps.s[j+1], mps[j+1], trunc)
		return err
	end
	err = 0.
	err2 = _move!(mps, i, j-1, trunc)
	err = max(err, err2)
	err2 = _move!(mps, k, j+1, trunc)
	err = max(err, err2)
	mps[j-1], mps.s[j], mps[j], mps.s[j+1], mps[j+1], err2 = bond_evolution3(m,
		mps.s[j-1], mps[j-1], mps.s[j], mps[j], mps.s[j+1], mps[j+1], trunc)
	err = max(err, err2)
	err2 = _move!(mps, j-1, i, trunc)
	err = max(err, err2)
	err2 = _move!(mps, j+1, k, trunc)
	err = max(err, err2)
	return err
end

function _apply_impl(key::Tuple{Int, Int, Int, Int}, m::AbstractTensorMap{<:Number, S, 4, 4}, mps, trunc::TruncationScheme) where S
	i, j, k, l = key
	if is_nn_pos(key)
		mps[j-1], mps.s[j], mps[j], mps.s[j+1], mps[j+1], mps.s[j+2], mps[j+2], err = bond_evolution4(
			m, mps.s[j-1], mps[j-1], mps.s[j], mps[j], mps.s[j+1], mps[j+1], mps.s[j+2], mps[j+2], trunc)
		return err
	end
	err = 0.
	err2 = _move!(mps, i, j-1, trunc)
	err = max(err, err2)
	err2 = _move!(mps, k, j+1, trunc)
	err = max(err, err2)
	err2 = _move!(mps, l, j+2, trunc)
	err = max(err, err2)	

	mps[j-1], mps.s[j], mps[j], mps.s[j+1], mps[j+1], mps.s[j+2], mps[j+2], err2 = bond_evolution4(m,
		mps.s[j-1], mps[j-1], mps.s[j], mps[j], mps.s[j+1], mps[j+1], mps.s[j+2], mps[j+2], trunc)
	err = max(err, err2)
	err2 = _move!(mps, j-1, i, trunc)
	err = max(err, err2)
	err2 = _move!(mps, j+2, l, trunc)
	err = max(err, err2)
	err2 = _move!(mps, j+1, k, trunc)
	err = max(err, err2)
	return err
end

DMRG.apply!(s::AbstractQuantumGate, mps::MPS; kwargs...) = _apply!(s, mps; kwargs...)
DMRG.apply!(s::AbstractQuantumGate, mps::InfiniteMPS; kwargs...) = _apply!(s, mps; kwargs...)
DMRG.apply!(circuit::AbstractQuantumCircuit, mps::MPS; kwargs...) = _apply!(circuit, mps; kwargs...)
DMRG.apply!(circuit::AbstractQuantumCircuit, mps::InfiniteMPS; kwargs...) = _apply!(circuit, mps; kwargs...)

function _apply!(s::AbstractQuantumGate, mps; trunc::TruncationScheme=DefaultTruncation) 
	(length(positions(s)) <= 4) || throw(ArgumentError("only 4-body (or less) gates are currently allowed"))
	svectors_uninitialized(mps) && canonicalize!(mps)
	_apply_impl(positions(s), op(s), mps, trunc)
	return mps
end 

function _apply!(circuit::AbstractQuantumCircuit, mps; kwargs...)
	for gate in circuit
		_apply!(gate, mps; kwargs...)
	end
	return mps
end
