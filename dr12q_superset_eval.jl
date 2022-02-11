### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 21d5a8d1-d087-4598-851f-8cb8e67fee83
begin
	using BSON, CSV, DataFrames, DelimitedFiles, FITSIO, Flux, HDF5, Printf, Statistics, StatsPlots
	include("Evaluation.jl")
	include("Neural.jl")
	include("Utils.jl")
	import .Evaluation, .Neural, .Utils
	Core.eval(Main, :(import Flux, NNlib))
end

# ╔═╡ ed4d438e-6aaa-11eb-051e-efe644cce631
md"""
# DR12Q Superset Evaluation
"""

# ╔═╡ 5887bc2f-44e4-4b75-bbdd-ffcb6f88a35e
function write_lst(id, filename)
	open("data/" * filename, "w") do file
		writedlm(file, sort(Utils.get_filename.(id[1, :], id[2, :], id[3, :])))
	end
end

# ╔═╡ 8643602a-66e9-11eb-3700-374047551428
begin
	dr12q_superset = h5open("data/dr12q_superset.hdf5", "r")
	# write filenames of spectra to files
	id_train = read(dr12q_superset, "id_tr")
	id_valid = read(dr12q_superset, "id_va")
	id_test = read(dr12q_superset, "id_te")
	write_lst(id_train, "dr12q_superset_train.lst")
	write_lst(id_valid, "dr12q_superset_valid.lst")
	write_lst(id_test, "dr12q_superset_test.lst")
	# load data
	X_valid = read(dr12q_superset, "X_va")
	X_test = read(dr12q_superset, "X_te")
	X_valid_gpu = gpu(X_valid)
	X_test_gpu = gpu(X_test)
	z_valid = read(dr12q_superset, "z_vi_va")
	# create DR12Q superset test DataFrame
	dr12q_superset_df = DataFrame(
		plate=id_test[1, :], mjd=id_test[2, :], fiberid=id_test[3, :],
		ẑ=read(dr12q_superset, "z_pred_te"),
		z_vi=read(dr12q_superset, "z_vi_te"),
		z_pipe=read(dr12q_superset, "z_pipe_te"))
	close(dr12q_superset)
	dr12q_superset_df
end

# ╔═╡ 39ac64c5-8841-478c-89b3-85a76d3d7c01
begin
	dr16q_superset = h5open("data/dr16q_superset.hdf5", "r")
	id_dr16q = read(dr16q_superset, "id")
	dr16q_superset_df = DataFrame(
		plate=id_dr16q[1, :], mjd=id_dr16q[2, :], fiberid=id_dr16q[3, :],
		z_qn_dr16q=read(dr16q_superset, "z_qn"),
		z_pca_dr16q=read(dr16q_superset, "z_pca"),
		z_pipe_dr16q=read(dr16q_superset, "z_pipe"))
	close(dr16q_superset)
	dr16q_superset_df
end

# ╔═╡ 9f42c9bd-2af6-4883-905d-5eb29699add8
superset_df = dropmissing(leftjoin(
	dr12q_superset_df, dr16q_superset_df, on=[:plate, :mjd, :fiberid]))

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
	reg_fc_gpu = gpu(BSON.load("models/reg_fc.bson")[:model])
	ẑ_valid_reg_fc = Neural.regress(reg_fc_gpu, X_valid_gpu)
	evaluate(z_valid, ẑ_valid_reg_fc)
end

# ╔═╡ 646c8844-d25a-4453-a4d9-e6f6279c183b
md"""
## Regression SZNet
"""

# ╔═╡ edd6e898-6797-11eb-2cee-791764fb425a
begin
	reg_sznet_gpu = gpu(BSON.load("models/reg_sznet.bson")[:model])
	ẑ_valid_reg_sznet = Neural.regress(reg_sznet_gpu, X_valid_gpu)
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
	clf_fc_gpu = gpu(BSON.load("models/clf_fc.bson")[:model])
	ẑ_valid_clf_fc = Neural.classify(clf_fc_gpu, X_valid_gpu)
	evaluate(z_valid, ẑ_valid_clf_fc)
end

# ╔═╡ c9c00f77-4f40-445c-b348-70754dfce19c
md"""
## Classification SZNet
"""

# ╔═╡ cd79f25b-36e0-463a-80b6-ebc622aa75d2
begin
	clf_sznet_gpu = gpu(BSON.load("models/clf_sznet.bson")[:model])
	ẑ_valid_clf_sznet = Neural.classify(clf_sznet_gpu, X_valid_gpu)
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
		model_gpu = gpu(BSON.load(path)[:model])
		ẑ_model = Neural.mcdropout(model_gpu, X_valid_gpu, T=20)
		push!(λs, 10.0 ^ exp)
		push!(cfrs, Evaluation.cfr(z_valid, ẑ_model))
	end
	scatter(λs, cfrs, xscale=:log, legend=:none)
end

# ╔═╡ f75d1c9a-46ef-411b-970c-6d4972a3f55b
begin
	bayes_sznet_gpu = gpu(BSON.load("models/bayes_sznet_1e-4.bson")[:model])
	ẑ_valid_bayes_sznet = Neural.mcdropout(bayes_sznet_gpu, X_valid_gpu, T=20)
	evaluate(z_valid, ẑ_valid_bayes_sznet)
end

# ╔═╡ 662ec364-2329-474e-8d6e-ff5b75f90ccc
md"## Evaluation on Test Set"

# ╔═╡ 3899a96f-e6d3-4802-acca-836f2a7bd71c
size(dr12q_superset_df, 1) - size(superset_df, 1)

# ╔═╡ dd5cec44-b54a-49d8-a535-b8f4b38729b7
evaluate(dr12q_superset_df.z_vi,dr12q_superset_df.ẑ)

# ╔═╡ 3cbc61c0-6d7a-46c9-a819-babba8690672
evaluate(dr12q_superset_df.z_vi, dr12q_superset_df.z_pipe)

# ╔═╡ 29bede71-3f12-4385-b4cc-ce4321811cde
evaluate(superset_df.z_vi, superset_df.z_pipe_dr16q)

# ╔═╡ f49227ae-5d16-43bd-b5fa-19e202e16b11
evaluate(superset_df.z_vi, superset_df.z_pca_dr16q)

# ╔═╡ a7261394-a8bb-431a-a2c2-2eea931e2b32
evaluate(superset_df.z_vi, superset_df.z_qn_dr16q)

# ╔═╡ 8f5bdb69-bd9c-4144-b297-f256729803ad
begin
	Δv_pred = Evaluation.computeΔv(dr12q_superset_df.z_vi, dr12q_superset_df.ẑ)
	Δv_pipe = Evaluation.computeΔv(dr12q_superset_df.z_vi, dr12q_superset_df.z_pipe)
	stop = max(maximum(Δv_pred), maximum(Δv_pipe))
	bins = range(0, stop=stop, length=128)
	plot(
		Δv_pred, seriestype=:stephist, bins=bins,
		yaxis=:log, label="Bayesian SZNet", xlabel="Δv", ylabel="Count")
	plot!(
		Δv_pipe, seriestype=:stephist, bins=bins,
		yaxis=:log, label="Pipeline")
end

# ╔═╡ cc01b586-df2b-4c43-ac60-5f35a9b50599
histogram2d(
	dr12q_superset_df.z_vi, dr12q_superset_df.ẑ, bins=Neural.N_LABELS,
	xlabel="Visual Redshift", ylabel="Predicted Redshift")

# ╔═╡ 1e27cb5f-8058-40c6-a02b-78ff8a9169c2
histogram2d(
	dr12q_superset_df.z_vi, dr12q_superset_df.z_pipe, bins=Neural.N_LABELS,
	xlabel="Visual Redshift", ylabel="Pipeline Redshift")

# ╔═╡ a723314b-86ff-408d-9349-bf8c0a7d2ef3
md"""
## Data for Cross-Match with DR16Q Superset
"""

# ╔═╡ 9d4ea5ec-b710-4bb7-899c-d047e18ad4a7
begin
	# read metadata of DR12Q superset
	dr12q_catalog_fits = FITS("data/Superset_DR12Q.fits")
	dr12q_catalog_df = DataFrame(
		plate=read(dr12q_catalog_fits[2], "PLATE"),
		mjd=read(dr12q_catalog_fits[2], "MJD"),
		fiberid=read(dr12q_catalog_fits[2], "FIBERID"),
		ra=read(dr12q_catalog_fits[2], "RA"),
		dec=read(dr12q_catalog_fits[2], "DEC"))
	close(dr12q_catalog_fits)
	# create DataFrame of train and validation identifiers
	id_xmatch = cat(id_train, id_valid, dims=2)
	xmatch_df = DataFrame(
		plate=id_xmatch[1, :], mjd=id_xmatch[2, :], fiberid=id_xmatch[3, :])
	xmatch_df = leftjoin(xmatch_df, dr12q_catalog_df, on=[:plate, :mjd, :fiberid])
	CSV.write("data/dr12q_superset_train_valid.csv", xmatch_df)
	xmatch_df
end

# ╔═╡ Cell order:
# ╟─ed4d438e-6aaa-11eb-051e-efe644cce631
# ╠═21d5a8d1-d087-4598-851f-8cb8e67fee83
# ╠═5887bc2f-44e4-4b75-bbdd-ffcb6f88a35e
# ╠═8643602a-66e9-11eb-3700-374047551428
# ╠═39ac64c5-8841-478c-89b3-85a76d3d7c01
# ╠═9f42c9bd-2af6-4883-905d-5eb29699add8
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
# ╠═f75d1c9a-46ef-411b-970c-6d4972a3f55b
# ╟─662ec364-2329-474e-8d6e-ff5b75f90ccc
# ╠═3899a96f-e6d3-4802-acca-836f2a7bd71c
# ╠═dd5cec44-b54a-49d8-a535-b8f4b38729b7
# ╠═3cbc61c0-6d7a-46c9-a819-babba8690672
# ╠═29bede71-3f12-4385-b4cc-ce4321811cde
# ╠═f49227ae-5d16-43bd-b5fa-19e202e16b11
# ╠═a7261394-a8bb-431a-a2c2-2eea931e2b32
# ╠═8f5bdb69-bd9c-4144-b297-f256729803ad
# ╠═cc01b586-df2b-4c43-ac60-5f35a9b50599
# ╠═1e27cb5f-8058-40c6-a02b-78ff8a9169c2
# ╟─a723314b-86ff-408d-9349-bf8c0a7d2ef3
# ╠═9d4ea5ec-b710-4bb7-899c-d047e18ad4a7
