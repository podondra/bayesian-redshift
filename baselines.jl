### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 8c5f26c2-5c7d-11eb-33b1-6db9019c828c
using HDF5

# ╔═╡ 2a8934bc-5c7e-11eb-0b71-a966b4b37054
include("evaluation.jl")

# ╔═╡ f3391cb6-5c7a-11eb-1267-db40587a741e
begin
	datafile = h5open("data/dr16q_superset.hdf5", "r")
	y_va = read(datafile, "z_va")
	y_pca_va = read(datafile, "z_pca_va")
	y_pipe_va = read(datafile, "z_pipe_va")
	y_qn_va = read(datafile, "z_qn_va")
end

# ╔═╡ 49d4f36c-5c7e-11eb-2ee2-6737c5d9cb2c
rmse(y_va, y_pca_va), catastrophic_redshift_ratio(y_va, y_pca_va)

# ╔═╡ 6eb55366-5c7e-11eb-272b-a13c9b77744b
rmse(y_va, y_pipe_va), catastrophic_redshift_ratio(y_va, y_pipe_va)

# ╔═╡ 787ec2c4-5c7e-11eb-1b55-850c597ba857
rmse(y_va, y_qn_va), catastrophic_redshift_ratio(y_va, y_qn_va)

# ╔═╡ Cell order:
# ╠═8c5f26c2-5c7d-11eb-33b1-6db9019c828c
# ╠═2a8934bc-5c7e-11eb-0b71-a966b4b37054
# ╠═f3391cb6-5c7a-11eb-1267-db40587a741e
# ╠═49d4f36c-5c7e-11eb-2ee2-6737c5d9cb2c
# ╠═6eb55366-5c7e-11eb-272b-a13c9b77744b
# ╠═787ec2c4-5c7e-11eb-1b55-850c597ba857
