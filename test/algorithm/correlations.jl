println("------------------------------------")
println("----|   Two-Tme Correlations  |-----")
println("------------------------------------")

function corr_hubbard_chain(L, J1, U, p)
	adag, pp, nn, JW = p["+"], p["++"], p["n↑n↓"], p["JW"]

	@tensor adagJW[1,2;3,5] := adag[1,2,3,4] * JW[4,5]

	terms = []
	for i in 1:L
		push!(terms, QTerm(i => nn, coeff=U))
	end
	for i in 1:L-1
		m = QTerm(i=>adagJW, i+1=>adag', coeff=-J1)
		push!(terms, m)
		push!(terms, m')
	end

	creation = QTerm(1=>JW, 2=>adag)
	annihilation = creation'

	ham = QuantumOperator(terms)
	return ham, prodmpo(physical_spaces(ham), creation), prodmpo(physical_spaces(ham), annihilation) 
end

function corr_initial_state_u1_su2(L)
	physpace = Rep[U₁×SU₂]((-0.5, 0)=>1, (0.5, 0)=>1, (0, 0.5)=>1)

	init_state = [(-0.5, 0) for i in 1:L]
	for i in 2:2:L
		init_state[i] = (0.5, 0)
	end
	n = sum([item[1] for item in init_state])

	right = Rep[U₁×SU₂]((n, 0)=>1)
	state = prodmps(ComplexF64, physpace, init_state, right=right )

	init_state = [(-0.5, 0) for i in 1:L]
	for i in 1:2:L
		init_state[i] = (0.5, 0)
	end
	n = sum([item[1] for item in init_state])

	right = Rep[U₁×SU₂]((n, 0)=>1)

	state = state + prodmps(ComplexF64, physpace, init_state, right=right )

	canonicalize!(state, alg = Orthogonalize(normalize=true))
	return state
end

function corr_initial_state_u1_u1(L)
	physpace = Rep[U₁×U₁]((0, 0)=>1, (0, 1)=>1, (1, 0)=>1, (1, 1)=>1)

	init_state = [(0, 0) for i in 1:L]
	for i in 2:2:L
		init_state[i] = (1, 1)
	end
	n = sum([item[1] for item in init_state])
	right = Rep[U₁×U₁]((n, n)=>1)
	state = prodmps(ComplexF64, physpace, init_state, right=right )

	init_state = [(0, 0) for i in 1:L]
	for i in 1:2:L
		init_state[i] = (1, 1)
	end
	n = sum([item[1] for item in init_state])
	right = Rep[U₁×U₁]((n, n)=>1)
	state = state + prodmps(ComplexF64, physpace, init_state, right=right )

	canonicalize!(state, alg = Orthogonalize(normalize=true))
	return state
end

@testset "Two-Tme correlations: general state" begin
	J = 1.
	U = 1.2
	tol = 1.0e-4
	# check general state correlations
	p = spinal_fermion_site_ops_u1_su2()

	ts = [0., 0.02, 0.05]
	stepsize = 0.01
	for L in (2, 4)
		h, sp_op, sm_op = corr_hubbard_chain(L, J, U, p)
		state = corr_initial_state_u1_su2(L)

		function corr_t(reverse::Bool)
			corr = correlation_2op_1t(h, sm_op, sp_op, copy(state), ts, stepper=TEBDStepper(stepsize=stepsize), reverse=reverse)
			return corr
		end

		function exact_corr_t(reverse::Bool)
			corr = exact_correlation_2op_1t(h, sm_op, sp_op, copy(state), ts, reverse=reverse)
			return corr
		end

		function corr_τ(reverse::Bool)
			corr = correlation_2op_1τ(h, sm_op, sp_op, copy(state), ts, stepper=TEBDStepper(stepsize=stepsize), reverse=reverse)
			return corr
		end

		function exact_corr_τ(reverse::Bool)
			corr = exact_correlation_2op_1τ(h, sm_op, sp_op, copy(state), ts, reverse=reverse)
			return corr
		end

		for m in (true, false)
			@test max_error(corr_t(m), exact_corr_t(m)) < tol
			@test max_error(corr_τ(m), exact_corr_τ(m)) < tol
		end
	end

end

function compute_gs(mpo, state)
	state_2 = copy(state)
	energies, delta = ground_state!(state_2, mpo, DMRG2(trunc=truncdimcutoff(D=20, ϵ=1.0e-8)))
	return energies[end], state_2
end

@testset "Two-Tme correlations: ground state" begin
	J = 1.
	U = 1.3
	tol = 1.0e-4
	p = spinal_fermion_site_ops_u1_su2()
	for L in (2, 4)
		h, sp_op, sm_op = corr_hubbard_chain(L, J, U, p)
		state = corr_initial_state_u1_su2(L)

		gs_energy, state = compute_gs(MPO(h), state)
		ts = [0., 0.02, 0.1]
		stepsize = 0.01

		function corr_t(reverse::Bool)
			corr = gs_correlation_2op_1t(h, sm_op, sp_op, copy(state), ts, gs_energy=gs_energy, stepper=TEBDStepper(stepsize=stepsize), reverse=reverse)
			return corr
		end

		function exact_corr_t(reverse::Bool)
			corr = exact_correlation_2op_1t(h, sm_op, sp_op, copy(state), ts, reverse=reverse)
			return corr
		end

		function corr_τ(reverse::Bool)
			corr = gs_correlation_2op_1τ(h, sm_op, sp_op, copy(state), ts, gs_energy=gs_energy, stepper=TEBDStepper(stepsize=stepsize), reverse=reverse)
			return corr
		end

		function exact_corr_τ(reverse::Bool)
			corr = exact_correlation_2op_1τ(h, sm_op, sp_op, copy(state), ts, reverse=reverse)
			return corr
		end

		for m in (true, false)
			@test max_error(corr_t(m), exact_corr_t(m)) < tol
			@test max_error(corr_τ(m), exact_corr_τ(m)) < tol

		end		
	end
end


