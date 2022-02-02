### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 21d5a8d1-d087-4598-851f-8cb8e67fee83
begin
	using BSON, DataFrames, DelimitedFiles, Flux, HDF5, Printf, Statistics, StatsPlots

	Core.eval(Main, :(import Flux, NNlib))

	include("Evaluation.jl")
	include("Neural.jl")
	include("Utils.jl")
	import .Evaluation
	import .Neural
	import .Utils
end

# ╔═╡ ed4d438e-6aaa-11eb-051e-efe644cce631
md"""
# DR12Q Superset Evaluation
"""

# ╔═╡ 39ac64c5-8841-478c-89b3-85a76d3d7c01
begin
	dr16q = h5open("data/dr16q_superset.hdf5", "r")
	id_dr16q = read(dr16q, "id")
	dr16q_df = DataFrame(
		plate=id_dr16q[1, :],
		mjd=id_dr16q[2, :],
		fiberid=id_dr16q[3, :],
		z_qn_dr16q=read(dr16q, "z_qn"),
		z_pca_dr16q=read(dr16q, "z_pca"),
		z_pipe_dr16q=read(dr16q, "z_pipe"))
	close(dr16q)
	dr16q_df
end

# ╔═╡ 8643602a-66e9-11eb-3700-374047551428
begin
	dr12q = h5open("data/dr12q_superset.hdf5", "r")
	id_train = read(dr12q, "id_tr")
	id_valid = read(dr12q, "id_va")
	id_test = read(dr12q, "id_te")

	X_valid = read(dr12q, "X_va") |> gpu
	X_test = read(dr12q, "X_te") |> gpu

	z_train = read(dr12q, "z_vi_tr")
	z_valid = read(dr12q, "z_vi_va")
	z_pipe_valid = read(dr12q, "z_pipe_va")

	dr12q_df = DataFrame(
		plate=id_test[1, :],
		mjd=id_test[2, :],
		fiberid=id_test[3, :],
		z_vi=read(dr12q, "z_vi_te"),
		z_pipe=read(dr12q, "z_pipe_te"))
	close(dr12q)
	test_df = dropmissing(leftjoin(dr12q_df, dr16q_df, on=[:plate, :mjd, :fiberid]))
end

# ╔═╡ aaae60cf-8275-488f-a98e-97dfe8a57890
open("data/dr12q_superset_train.lst", "w") do file
	filenames_train = Utils.get_filename.(id_train[1, :], id_train[2, :], id_train[3, :])
	writedlm(file, sort(filenames_train))
end

# ╔═╡ 0a862141-d497-47e4-90c6-b36756f3b76c
open("data/dr12q_superset_valid.lst", "w") do file
	filenames_valid = Utils.get_filename.(id_valid[1, :], id_valid[2, :], id_valid[3, :])
	writedlm(file, sort(filenames_valid))
end

# ╔═╡ 3288aeab-05c2-47c3-bd31-070bf0dbc43f
open("data/dr12q_superset_test.lst", "w") do file
	filenames_test = Utils.get_filename.(id_test[1, :], id_test[2, :], id_test[3, :])
	writedlm(file, sort(filenames_test))
end

# ╔═╡ f167bf89-515c-42fb-ad27-3548e6d1ad0a
function evaluate(z, ẑ)
	Dict(
		"rmse" => Evaluation.rmse(z, ẑ),
		"mean Δv" => Evaluation.meanΔv(z, ẑ),
		"std Δv" => std(Evaluation.computeΔv(z, ẑ)),
		"median Δv" => Evaluation.medianΔv(z, ẑ),
		"cfr" => Evaluation.cfr(z, ẑ))
end

# ╔═╡ aae3e861-1c60-48c1-b070-3a40e712885c
md"""
## Regression FCNN
"""

# ╔═╡ 77becfae-894d-46fd-b0b2-b09e2e2a0a17
begin
	reg_fc = BSON.load("models/reg_fc.bson")[:model] |> gpu
	ẑ_valid_reg_fc = Neural.regress(reg_fc, X_valid)
	evaluate(z_valid, ẑ_valid_reg_fc)
end

# ╔═╡ 646c8844-d25a-4453-a4d9-e6f6279c183b
md"""
## Regression SZNet
"""

# ╔═╡ edd6e898-6797-11eb-2cee-791764fb425a
begin
	reg_sznet = BSON.load("models/reg_sznet.bson")[:model] |> gpu
	ẑ_valid_reg_sznet = Neural.regress(reg_sznet, X_valid)
	evaluate(z_valid, ẑ_valid_reg_sznet)
end

# ╔═╡ e6d5d6fa-6aab-11eb-0434-19c6a5a97099
begin
	density(z_valid, label="Visual Redshift", xlabel="Redshift", ylabel="Density")
	density!(ẑ_valid_reg_sznet, label="Regression SZNet")
end

# ╔═╡ a54b57cf-2d52-4029-b69a-09d5cc62ed00
md"""
## Classfication FCNN
"""

# ╔═╡ 2ccf27f6-5da0-463b-a6c7-f7f5711825ef
begin
	clf_fc = BSON.load("models/clf_fc.bson")[:model] |> gpu
	ẑ_valid_clf_fc = Neural.classify(clf_fc, X_valid)
	evaluate(z_valid, ẑ_valid_clf_fc)
end

# ╔═╡ c9c00f77-4f40-445c-b348-70754dfce19c
md"""
## Classification SZNet
"""

# ╔═╡ cd79f25b-36e0-463a-80b6-ebc622aa75d2
begin
	clf_sznet = BSON.load("models/clf_sznet.bson")[:model] |> gpu
	ẑ_valid_clf_sznet = Neural.classify(clf_sznet, X_valid)
	evaluate(z_valid, ẑ_valid_clf_sznet)
end

# ╔═╡ e1ada0e9-c547-4642-92e6-71a8ef1ce5ad
begin
	density(z_valid, label="Visual Redshift", xlabel="Redshift", ylabel="Density")
	density!(ẑ_valid_clf_sznet, label="Classification SZNet")
end

# ╔═╡ bbf6a903-5abc-4faf-8963-b7824623a324
md"## Grid Search for ``\lambda``"

# ╔═╡ 22a7b903-4267-439d-b469-37a80c53ddee
begin
	λs, cfrs = [], []
	for exp in -7:-2
		path = @sprintf("models/bayes_sznet_1e%d.bson", exp)
		model = BSON.load(path)[:model] |> gpu
		ẑ_model = Neural.mcdropout(model, X_valid, T=20)
		push!(λs, 10.0 ^ exp)
		push!(cfrs, Evaluation.cfr(z_valid, ẑ_model))
	end
	cfrs
end

# ╔═╡ 92475d81-590b-4b0f-a53b-4fe9cabf8881
scatter(λs, cfrs, xscale=:log, legend=:none)

# ╔═╡ 662ec364-2329-474e-8d6e-ff5b75f90ccc
md"## Evaluation on Test Set"

# ╔═╡ 3899a96f-e6d3-4802-acca-836f2a7bd71c
size(dr12q_df, 1), size(test_df, 1), size(dr12q_df, 1) - size(test_df, 1)

# ╔═╡ 3cbc61c0-6d7a-46c9-a819-babba8690672
evaluate(dr12q_df.z_vi, dr12q_df.z_pipe)

# ╔═╡ 29bede71-3f12-4385-b4cc-ce4321811cde
evaluate(test_df.z_vi, test_df.z_pipe_dr16q)

# ╔═╡ f49227ae-5d16-43bd-b5fa-19e202e16b11
evaluate(test_df.z_vi, test_df.z_pca_dr16q)

# ╔═╡ a7261394-a8bb-431a-a2c2-2eea931e2b32
evaluate(test_df.z_vi, test_df.z_qn_dr16q)

# ╔═╡ dd5cec44-b54a-49d8-a535-b8f4b38729b7
begin
	bayes_sznet = BSON.load("models/bayes_sznet_1e-4.bson")[:model] |> gpu
	ẑ_test = Neural.mcdropout(bayes_sznet, X_test, T=20)
	evaluate(dr12q_df[!, :z_vi], ẑ_test)
end

# ╔═╡ 8f5bdb69-bd9c-4144-b297-f256729803ad
begin
	Δv = Evaluation.computeΔv(dr12q_df.z_vi, ẑ_test)
	Δv_pipe = Evaluation.computeΔv(dr12q_df.z_vi, dr12q_df.z_pipe)
	stop = max(maximum(Δv), maximum(Δv_pipe))
	bins = range(0, stop=stop, length=128)
	plot(Δv_pipe, seriestype=:stephist, bins=bins, yaxis=:log, label="Pipeline", xlabel="Δv", ylabel="Count")
	plot!(Δv, seriestype=:stephist, bins=bins, yaxis=:log, label="Bayesian SZNet")
end

# ╔═╡ cc01b586-df2b-4c43-ac60-5f35a9b50599
histogram2d(dr12q_df[!, :z_vi], ẑ_test, bins=Neural.N_LABELS, xlabel="Visual Redshift", ylabel="Predicted Redshift")

# ╔═╡ 1e27cb5f-8058-40c6-a02b-78ff8a9169c2
histogram2d(dr12q_df[!, :z_vi], dr12q_df[!, :z_pipe], bins=Neural.N_LABELS, xlabel="Visual Redshift", ylabel="Pipeline Redshift")

# ╔═╡ Cell order:
# ╟─ed4d438e-6aaa-11eb-051e-efe644cce631
# ╠═21d5a8d1-d087-4598-851f-8cb8e67fee83
# ╠═39ac64c5-8841-478c-89b3-85a76d3d7c01
# ╠═8643602a-66e9-11eb-3700-374047551428
# ╠═aaae60cf-8275-488f-a98e-97dfe8a57890
# ╠═0a862141-d497-47e4-90c6-b36756f3b76c
# ╠═3288aeab-05c2-47c3-bd31-070bf0dbc43f
# ╠═f167bf89-515c-42fb-ad27-3548e6d1ad0a
# ╟─aae3e861-1c60-48c1-b070-3a40e712885c
# ╠═77becfae-894d-46fd-b0b2-b09e2e2a0a17
# ╟─646c8844-d25a-4453-a4d9-e6f6279c183b
# ╠═edd6e898-6797-11eb-2cee-791764fb425a
# ╠═e6d5d6fa-6aab-11eb-0434-19c6a5a97099
# ╟─a54b57cf-2d52-4029-b69a-09d5cc62ed00
# ╠═2ccf27f6-5da0-463b-a6c7-f7f5711825ef
# ╟─c9c00f77-4f40-445c-b348-70754dfce19c
# ╠═cd79f25b-36e0-463a-80b6-ebc622aa75d2
# ╠═e1ada0e9-c547-4642-92e6-71a8ef1ce5ad
# ╟─bbf6a903-5abc-4faf-8963-b7824623a324
# ╠═22a7b903-4267-439d-b469-37a80c53ddee
# ╠═92475d81-590b-4b0f-a53b-4fe9cabf8881
# ╟─662ec364-2329-474e-8d6e-ff5b75f90ccc
# ╠═3899a96f-e6d3-4802-acca-836f2a7bd71c
# ╠═3cbc61c0-6d7a-46c9-a819-babba8690672
# ╠═29bede71-3f12-4385-b4cc-ce4321811cde
# ╠═f49227ae-5d16-43bd-b5fa-19e202e16b11
# ╠═a7261394-a8bb-431a-a2c2-2eea931e2b32
# ╠═dd5cec44-b54a-49d8-a535-b8f4b38729b7
# ╠═8f5bdb69-bd9c-4144-b297-f256729803ad
# ╠═cc01b586-df2b-4c43-ac60-5f35a9b50599
# ╠═1e27cb5f-8058-40c6-a02b-78ff8a9169c2
