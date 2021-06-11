### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 21d5a8d1-d087-4598-851f-8cb8e67fee83
begin
	using BSON, DataFrames, FITSIO, Flux, HDF5, Printf, StatsBase, StatsPlots
	include("Evaluation.jl"); using .Evaluation
	include("Neural.jl"); using .Neural
	include("Utils.jl"); using .Utils
	Core.eval(Main, :(import Flux, NNlib))
end

# ╔═╡ ed4d438e-6aaa-11eb-051e-efe644cce631
md"# SDSS DR12Q Superset Evaluation"

# ╔═╡ 39ac64c5-8841-478c-89b3-85a76d3d7c01
begin
	dr16q_file = h5open("data/dr16q_superset.hdf5", "r")
	id_dr16q = read(dr16q_file, "id")
	dr16q_df = DataFrame(
		plate=id_dr16q[1, :],
		mjd=id_dr16q[2, :],
		fiberid=id_dr16q[3, :],
		z_qn=read(dr16q_file, "z_qn"),
		z_pca=read(dr16q_file, "z_pca"),
		z_pipe_dr16q=read(dr16q_file, "z_pipe"))
	close(dr16q_file)
	dr16q_df
end

# ╔═╡ 8643602a-66e9-11eb-3700-374047551428
begin
	dr12q_file = h5open("data/dr12q_superset.hdf5", "r")
	y_train = read(dr12q_file, "z_vi_tr")

	id_valid = read(dr12q_file, "id_va")
	X_valid = read(dr12q_file, "X_va") |> gpu
	y_valid = read(dr12q_file, "z_vi_va")
	y_pipe_valid = read(dr12q_file, "z_pipe_va")

	id_test = read(dr12q_file, "id_te")
	X_test = read(dr12q_file, "X_te") |> gpu
	dr12q_df = DataFrame(
		plate=id_test[1, :],
		mjd=id_test[2, :],
		fiberid=id_test[3, :],
		z_vi=read(dr12q_file, "z_vi_te"),
		z_pipe=read(dr12q_file, "z_pipe_te"))

	close(dr12q_file)
	test_df = dropmissing(leftjoin(dr12q_df, dr16q_df, on=[:plate, :mjd, :fiberid]))
end

# ╔═╡ 646c8844-d25a-4453-a4d9-e6f6279c183b
md"## Regression Model"

# ╔═╡ edd6e898-6797-11eb-2cee-791764fb425a
begin
	reg_model = BSON.load("models/regression_model.bson")[:model] |> gpu
	ŷ_reg = Neural.regress(reg_model, X_valid) |> cpu
	reg_model
end

# ╔═╡ dcfade4f-12c3-4aa3-97e1-fb4c63f72cb4
Evaluation.rmse(y_valid, ŷ_reg),
Evaluation.median_Δv(y_valid, ŷ_reg),
Evaluation.cat_z_ratio(y_valid, ŷ_reg)

# ╔═╡ e6d5d6fa-6aab-11eb-0434-19c6a5a97099
begin
	density(y_valid, label="Visual z", xlabel="z", ylabel="Density")
	density!(ŷ_reg, label="Regression z")
end

# ╔═╡ 5f25ed6f-ef8d-49c0-993d-e3d5c445fd99
scatter(y_valid, ŷ_reg, legend=:none, xlabel="Visual", ylabel="Regression")

# ╔═╡ 1d52576a-6abb-11eb-2dd8-c388b2365ddd
begin	
	Δv_reg = Evaluation.compute_Δv(y_valid, ŷ_reg)
	# random
	i = rand(1:size(id_valid, 2))
	# cat. z
	i = rand((1:size(id_valid, 2))[Δv_reg .> 3000])
	# absolute error
	i = sortperm(abs.(y_valid - ŷ_reg))[end]

	Utils.plot_spectrum(
		X_valid[:, i], legend=:none,
		title=@sprintf(
			"z = %f; ẑ = %f; Δv = %f",
			y_valid[i], ŷ_reg[i], Δv_reg[i]))
	Utils.plot_spectral_lines!(y_valid[i])
	regression_plot = Utils.plot_spectral_lines!(
		ŷ_reg[i], color=:red, location=:bottom)
	plot(
		regression_plot,
		plot(Utils.get_linear_spectrum(
				"Superset_DR12Q", id_valid[:, i]...)..., legend=:none),
		layout=@layout [a; b])
end

# ╔═╡ c9c00f77-4f40-445c-b348-70754dfce19c
md"## Classification Model"

# ╔═╡ 734bf33f-3f44-42d3-9c69-98b111e5495b
minimum(y_train), maximum(y_train), length(0:0.01:6.44)

# ╔═╡ cd79f25b-36e0-463a-80b6-ebc622aa75d2
begin
	clf_model = BSON.load("models/classification_model.bson")[:model] |> gpu
	ŷ_clf = Neural.classify(clf_model, X_valid) |> cpu
	clf_model
end

# ╔═╡ f8ec4620-fc02-4dca-b005-0961de8ed1af
Evaluation.rmse(y_valid, ŷ_clf),
Evaluation.median_Δv(y_valid, ŷ_clf),
Evaluation.cat_z_ratio(y_valid, ŷ_clf)

# ╔═╡ e1ada0e9-c547-4642-92e6-71a8ef1ce5ad
begin
	density(y_valid, label="Visual z", xlabel="z", ylabel="Density")
	density!(ŷ_clf, label="Classification z")
end

# ╔═╡ 8d64cf38-8af9-4b35-89b2-daa60fdab019
scatter(y_valid, ŷ_clf, legend=:none, xlabel="Visual", ylabel="Classification")

# ╔═╡ 8c3f2374-3c9e-4fa5-a685-f2026eb38476
begin
	Δv_clf = Evaluation.compute_Δv(y_valid, ŷ_clf)
	# random
	j = rand(1:size(id_valid, 2))
	# cat. z
	j = rand((1:size(id_valid, 2))[Δv_clf .> 3000])
	# absolute error
	j = sortperm(abs.(y_valid - ŷ_clf))[end]

	Utils.plot_spectrum(
		X_valid[:, j], legend=:none,
		title=@sprintf(
			"z = %f; ẑ = %f; Δv = %f",
			y_valid[j], ŷ_clf[j], Δv_clf[j]))
	Utils.plot_spectral_lines!(y_valid[j])
	clf_plot = Utils.plot_spectral_lines!(
		ŷ_clf[j], color=:red, location=:bottom)
	plot(
		clf_plot,
		plot(Utils.get_linear_spectrum(
				"Superset_DR12Q", id_valid[:, j]...)..., legend=:none),
		layout=@layout [a; b])
end

# ╔═╡ bbf6a903-5abc-4faf-8963-b7824623a324
md"## Grid Search"

# ╔═╡ 22a7b903-4267-439d-b469-37a80c53ddee
begin
	df = DataFrame(λ=[], rmse=[], median_Δv=[], cat_z_ratio=[])
	exps = -10:-2
	for exp in exps
		model_path = @sprintf "models/classification_1e%d.bson" exp
		model = BSON.load(model_path)[:model] |> gpu
		ŷ = Neural.classify(model, X_valid) |> cpu
		push!(df, [
				10.0 ^ exp,
				Evaluation.rmse(y_valid, ŷ),
				Evaluation.median_Δv(y_valid, ŷ),
				Evaluation.cat_z_ratio(y_valid, ŷ)])
	end
	df
end

# ╔═╡ 92475d81-590b-4b0f-a53b-4fe9cabf8881
scatter(df[:λ], df[:cat_z_ratio], xscale=:log, yscale=:log, legend=:none)

# ╔═╡ fe9bf8ee-4701-4e10-937e-a9659fc1ea53
md"## Baselines"

# ╔═╡ 3cbc61c0-6d7a-46c9-a819-babba8690672
Evaluation.rmse(dr12q_df[:z_vi], dr12q_df[:z_pipe]),
Evaluation.median_Δv(dr12q_df[:z_vi], dr12q_df[:z_pipe]),
Evaluation.cat_z_ratio(dr12q_df[:z_vi], dr12q_df[:z_pipe])

# ╔═╡ 29bede71-3f12-4385-b4cc-ce4321811cde
Evaluation.rmse(test_df[:z_vi], test_df[:z_pipe_dr16q]),
Evaluation.median_Δv(test_df[:z_vi], test_df[:z_pipe_dr16q]),
Evaluation.cat_z_ratio(test_df[:z_vi], test_df[:z_pipe_dr16q])

# ╔═╡ f49227ae-5d16-43bd-b5fa-19e202e16b11
Evaluation.rmse(test_df[:z_vi], test_df[:z_pca]),
Evaluation.median_Δv(test_df[:z_vi], test_df[:z_pca]),
Evaluation.cat_z_ratio(test_df[:z_vi], test_df[:z_pca])

# ╔═╡ a7261394-a8bb-431a-a2c2-2eea931e2b32
Evaluation.rmse(test_df[:z_vi], test_df[:z_qn]),
Evaluation.median_Δv(test_df[:z_vi], test_df[:z_qn]),
Evaluation.cat_z_ratio(test_df[:z_vi], test_df[:z_qn])

# ╔═╡ 662ec364-2329-474e-8d6e-ff5b75f90ccc
md"## Evaluation on Test Set"

# ╔═╡ dd5cec44-b54a-49d8-a535-b8f4b38729b7
begin
	final_model = BSON.load("models/classification_1e-4.bson")[:model] |> gpu
	ŷ_test = Neural.classify(final_model, X_test) |> cpu
end

# ╔═╡ 4008b28a-595d-481e-ad8b-7e8b3137124d
Evaluation.rmse(dr12q_df[:z_vi], ŷ_test),
Evaluation.median_Δv(dr12q_df[:z_vi], ŷ_test),
Evaluation.cat_z_ratio(dr12q_df[:z_vi], ŷ_test)

# ╔═╡ Cell order:
# ╟─ed4d438e-6aaa-11eb-051e-efe644cce631
# ╠═21d5a8d1-d087-4598-851f-8cb8e67fee83
# ╠═39ac64c5-8841-478c-89b3-85a76d3d7c01
# ╠═8643602a-66e9-11eb-3700-374047551428
# ╟─646c8844-d25a-4453-a4d9-e6f6279c183b
# ╠═edd6e898-6797-11eb-2cee-791764fb425a
# ╠═dcfade4f-12c3-4aa3-97e1-fb4c63f72cb4
# ╠═e6d5d6fa-6aab-11eb-0434-19c6a5a97099
# ╠═5f25ed6f-ef8d-49c0-993d-e3d5c445fd99
# ╠═1d52576a-6abb-11eb-2dd8-c388b2365ddd
# ╟─c9c00f77-4f40-445c-b348-70754dfce19c
# ╠═734bf33f-3f44-42d3-9c69-98b111e5495b
# ╠═cd79f25b-36e0-463a-80b6-ebc622aa75d2
# ╠═f8ec4620-fc02-4dca-b005-0961de8ed1af
# ╠═e1ada0e9-c547-4642-92e6-71a8ef1ce5ad
# ╠═8d64cf38-8af9-4b35-89b2-daa60fdab019
# ╠═8c3f2374-3c9e-4fa5-a685-f2026eb38476
# ╟─bbf6a903-5abc-4faf-8963-b7824623a324
# ╠═22a7b903-4267-439d-b469-37a80c53ddee
# ╠═92475d81-590b-4b0f-a53b-4fe9cabf8881
# ╟─fe9bf8ee-4701-4e10-937e-a9659fc1ea53
# ╠═3cbc61c0-6d7a-46c9-a819-babba8690672
# ╠═29bede71-3f12-4385-b4cc-ce4321811cde
# ╠═f49227ae-5d16-43bd-b5fa-19e202e16b11
# ╠═a7261394-a8bb-431a-a2c2-2eea931e2b32
# ╟─662ec364-2329-474e-8d6e-ff5b75f90ccc
# ╠═dd5cec44-b54a-49d8-a535-b8f4b38729b7
# ╠═4008b28a-595d-481e-ad8b-7e8b3137124d
