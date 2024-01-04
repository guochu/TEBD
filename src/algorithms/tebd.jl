function _expm(x::QuantumOperator{S}, dt::Number) where {S <: ElementarySpace}
	isconstant(x) || throw(ArgumentError("only constant operator can be exponentiated"))
	data = todict(x)
	r = QuantumCircuit{S}()
	for (k, v) in data
		@assert !isempty(v)
		m = _join(v[1][1]) * scalar(v[1][2])
		for i in 2:length(v)
			m += _join(v[i][1]) * scalar(v[i][2])

		end
		if !iszero(m)
			push!(r, QuantumGate(k, exp!(m * dt)))
		end
	end	
	return r
end



function _join(v::Vector{<:MPOTensor})
	isempty(v) && throw(ArgumentError())
	util = GeneralHamiltonians.get_trivial_leg(v[1])
	if length(v) == 1
		@tensor r[-1 ; -2] := conj(util[1]) * v[1][1,-1,2,-2] * util[2]
	elseif length(v) == 2
		@tensor r[-1 -2; -3 -4] := conj(util[1]) * v[1][1,-1,2,-3] * v[2][2,-2,3,-4] * util[3]
	elseif length(v) == 3
		@tensor r[-1 -2 -3; -4 -5 -6] := conj(util[1]) * v[1][1,-1,2,-4] * v[2][2,-2,3,-5] * v[3][3,-3,4,-6] * util[4]
	elseif length(v) == 4
		@tensor r[-1 -2 -3 -4; -5 -6 -7 -8] := conj(util[1]) * v[1][1,-1,2,-5] * v[2][2,-2,3,-6] * v[3][3,-3,4,-7] * v[4][4,-4,5,-8] * util[5]
	else
		throw(ArgumentError("only support up to 4-body terms"))
	end
	return r
end

is_lower_than(ks, L::Int) = all(x -> length(x) < L, ks)
function _is_nn_single(key)
	(length(key) != 2) && error("input should be a tuple of 2")
	return key[1]+1 == key[2]
end

is_nn(ks) = all(_is_nn_single, ks)

function split_nn_ham(ham::QuantumOperator)
	is_nn(keys(ham)) || error("splt nn requires a nearest neighbour hamiltonian")
	ham_even = typeof(ham)(physical_spaces(ham))
	ham_odd = typeof(ham)(physical_spaces(ham))
	for item in qterms(ham)
		@assert length(positions(item)) == 1
		i, j = positions(item)
		(j == i+1) || error("hamiltonian contains non-nearest-neighbour term")
		if i%2==0
			push!(ham_even, item)
		else
			push!(ham_odd, item)
		end
	end
	return ham_even, ham_odd
end

function _expm(ham, t::Number, dt::Number)
	if isconstant(ham)
		return _expm(ham, dt)
	else
		return _expm(ham(t), dt)
	end
end 

# function _nn_tebd2order(ham_A, ham_B, dt)
# 	mpoevenhalf = _expm(ham_A, dt/2)
# 	mpooddone = _expm(ham_B, dt)
# 	circuit = similar(mpoevenhalf)
# 	append!(circuit, mpoevenhalf)
# 	append!(circuit, mpooddone)
# 	append!(circuit, mpoevenhalf)
# 	return circuit
# end

# function _compute_alpha_beta()
# 	ttt = 2^(1/3)
# 	beta1 = 1/(2-ttt)
# 	alpha1 = beta1/2
# 	alpha2 = ((1-ttt)/2)*beta1
# 	beta2 = -ttt*beta1
# 	return alpha1, alpha2, beta1, beta2
# end

# function _nn_tebd4order(ham_A, ham_B, dt)
# 	is_nn(Base.keys(ham)) || error("tebd2order requires a nearest neighbour hamiltonian.")
# 	alpha1, alpha2, beta1, beta2 = _compute_alpha_beta()


# 	# mpoevenbeta1 = _expm(ham_A, beta1 * dt)
# 	mpoevenbeta1half = _expm(ham_A, beta1 * dt / 2)
# 	mpoevenalpha2 = _expm(ham_A, alpha2 * dt)
# 	mpooddbeta1 = _expm(ham_B, beta1 * dt)
# 	mpooddbeta2 = _expm(ham_B, beta2 * dt)

# 	circuit = similar(mpoevenbeta1half)
# 	append!(circuit, mpoevenbeta1half)
# 	append!(circuit, mpooddbeta1)
# 	append!(circuit, mpoevenalpha2)
# 	append!(circuit, mpooddbeta2)
# 	append!(circuit, mpoevenalpha2)
# 	append!(circuit, mpooddbeta1)
# 	append!(circuit, mpoevenbeta1half)
# 	return circuit
# end

# function _generic_tebd2order(ham, dt)
# 	tmp = _expm(ham, dt/2)
# 	circuit = similar(tmp)
# 	append!(circuit, tmp)
# 	append!(circuit, reverse(tmp))
# 	return circuit
# end

_td_tebd1order(ham, t, dt) = _expm(ham, t+dt, dt)

function _td_generic_tebd2order(ham, t, dt)
	tmp = _expm(ham, t+dt/2, dt/2)
	circuit = similar(tmp)
	append!(circuit, tmp)
	append!(circuit, reverse(tmp))
	return circuit
end


"""
	used for nearest neighbour two body hamiltonian
"""
function _td_nn_tebd2order(ham_A, ham_B, t, dt)
	mpoevenhalf = _expm(ham_A, t+dt/2, dt/2)
	mpooddone = _expm(ham_B, t+dt/2, dt)
	circuit = similar(mpoevenhalf)
	append!(circuit, mpoevenhalf)
	append!(circuit, mpooddone)
	append!(circuit, mpoevenhalf)
	return circuit
end


_propagator_AB_2order(ham_A, ham_B, tf::Number, ti::Number) = _td_nn_tebd2order(
	ham_A, ham_B, ti, tf-ti)
# _propagator_AB_2order(ham_A, ham_B, dt::Number) = _nn_tebd2order(ham_A, ham_B, dt)
_propagator_generic_2order(ham, tf::Number, ti::Number) = _td_generic_tebd2order(ham, ti, tf-ti)
# _propagator_generic_2order(ham, dt::Number) = _generic_tebd2order(ham, dt)

function propagator_2order(ham::QuantumOperator, tf::Number, ti::Number)
	ks = keys(ham)
	if is_lower_than(ks, 2)
		if is_nn(ks)
		    ham_A, ham_B = split_nn_ham(ham)
		    return _propagator_AB_2order(ham_A, ham_B, tf, ti)
		end
	end
	return _propagator_generic_2order(ham, tf, ti)
end

# function propagator_2order(ham::QuantumOperator, dt::Number)
# 	ks = keys(storage(ham))
# 	if is_lower_than(ks, 2)
# 		if is_nn(ks)
# 		    ham_A, ham_B = split_nn_ham(ham)
# 		    return _propagator_AB_2order(ham_A, ham_B, dt)
# 		end
# 	end
# 	return _propagator_generic_2order(ham, dt)
# end


function _propagator_impl(ham::QuantumOperator, tf::Number, ti::Number, order::Int=1)
	(order == 1) && return propagator_2order(ham, tf, ti)
	# circuit = GenericCircuit1D()
	p = order - 1	
	sp = 1/(4 - 4^(1/(2*p+1)))
	dt = tf - ti
	circuit = _propagator_impl(ham, ti + sp * dt, ti, p)
	append!(circuit, _propagator_impl(ham, ti + 2 * sp * dt, ti + sp * dt, p) )
	append!(circuit, _propagator_impl(ham, ti + (1 - 2*sp) * dt, ti + 2 * sp * dt, p) ) 
	append!(circuit, _propagator_impl(ham, ti + (1-sp) * dt, ti + (1-2*sp) * dt, p) ) 
	append!(circuit, _propagator_impl(ham, ti + dt, ti + (1-sp) * dt, p) ) 
	# return fuse_gates(circuit)
	return circuit
end
	
"""
	trotter_propagator(ham::QuantumOperator, tspan::Tuple{<:Number, <:Number}; order::Int=2, dt::Number=0.01)

Trotter expansion to order 'order', with stepsize 'dt'. 

'tspan' is a tuple specifying the starting time and ending time for the time evolution, 
namely tspan = (t_start, t_end)

'ham' could either be a Hamiltonian or a Super Hamiltonian, and could be time-dependent 
with 'isconstant(ham)=false'

This function is also used for infinite TEBD, in which case the order is really a fake order

algorithm reference: "Higher order decompositions of ordered operator exponentials"
"""
trotter_propagator(ham::QuantumOperator, tspan::Tuple{<:Number, <:Number}; kwargs...) = _trotter_propagator(absorb_one_bodies(ham), tspan; kwargs...)
trotter_propagator(ham::QuantumOperator, dt::Number; kwargs...) = trotter_propagator(ham, (0, dt); kwargs...)

function _trotter_propagator(ham::QuantumOperator, tspan::Tuple{<:Number, <:Number}; order::Int=2, stepsize::Number=0.01)
	p = div(order, 2)
	(p * 2 == order) || throw(ArgumentError("only even order supported")) 
	dt = stepsize
	ti, tf = tspan
	tdiff = tf - ti
	nsteps = round(Int, abs(tdiff / dt)) 	
	if nsteps == 0
		nsteps = 1
	end
	dt = tdiff / nsteps
	local circuit
	for i in 1:nsteps
		if @isdefined circuit
			append!(circuit, _propagator_impl(ham, ti + i * dt, ti + (i-1)*dt, p))
		else
			circuit = _propagator_impl(ham, ti + i * dt, ti + (i-1)*dt, p)
		end
	end
	# return fuse_gates(circuit)
	return circuit
end
