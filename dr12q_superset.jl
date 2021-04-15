### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ 2a2861e4-9d00-11eb-08fd-972e78fd9d18
using DataFrames, FITSIO, HDF5, Statistics, StatsPlots

# ╔═╡ 24ee40d4-7e63-4654-b112-b7ecb7b912d2
md"# DR12Q Superset Catalogue

See the section Previously released SDSS quasar catalogs on [The Sloan Digital Sky Survey Quasar Catalog: fourteenth data release](https://www.sdss.org/dr14/algorithms/qso_catalog/).
QSOs catalogue are [on the SAS](https://data.sdss.org/sas/dr14/eboss/qso/).
Data model of the `Superset_DR12Q.fits` file is on [data.sdss.org](https://data.sdss.org/datamodel/files/BOSS_QSO/DR12Q/DR12Q_superset.html)."

# ╔═╡ 6779d063-2ad5-4a09-9bb8-22cb97ba711d
begin
	superset_fits = FITS("data/Superset_DR12Q.fits")
	superset = DataFrame(
		plate=read(superset_fits[2], "PLATE"),
		mjd=read(superset_fits[2], "MJD"),
		fiberid=read(superset_fits[2], "FIBERID"),
		z_vi=read(superset_fits[2], "Z_VI"),
		z_pipe=read(superset_fits[2], "Z_PIPE"),
		class_person=read(superset_fits[2], "CLASS_PERSON"),
		z_conf_person=read(superset_fits[2], "Z_CONF_PERSON"))
end

# ╔═╡ 43c3d594-e3a1-4af1-b093-f32fcd5b7e70
@df superset histogram(:z_vi, xlabel="z", ylabel="Count", legend=:none)

# ╔═╡ d93bd116-4d8e-42cb-a27c-9a7ee06d67dd
md"There are redshifts smaller than 0."

# ╔═╡ eb734e4c-0a9c-43c0-a9ff-fe1fab245783
begin
	lt_zero_idx = superset[:z_vi] .< 0.0
	sum(lt_zero_idx), unique(superset[lt_zero_idx, :z_vi])
end

# ╔═╡ 0bfe5e2b-8095-412d-85f3-aa373ab20cc9
md"`CLASS_PERSON` with value 1 or 4 and `Z_CONF_PERSON` with value 3 is `Star`, resp. `Galaxy`."

# ╔═╡ 541e307d-daa7-492c-95b4-6f835252d4d8
begin
	gt_minus_one_idx = -1.0 .< superset[:z_vi]
	superset[gt_minus_one_idx .& lt_zero_idx, :]
end

# ╔═╡ 7ffb928d-cdff-4eb6-aaf0-86a132f23572
md"For now leave them aside. It is small amout (6743) versus full superset (546856)."

# ╔═╡ 43f3be7c-aada-4e66-aee8-33abecabe7e8
subset = superset[.~lt_zero_idx, :]

# ╔═╡ 83889e5c-e5d3-4afb-a726-c5ee7c7e7969
md"## Wavelength Range

We need `specObj-dr12.fits` file available [on the SAS](https://data.sdss.org/sas/dr12/sdss/spectro/redux/).
From [data model](https://data.sdss.org/datamodel/files/SPECTRO_REDUX/specObj.html): `WAVEMIN` is 'Minimum observed (vacuum) wavelength (Angstroms)';
and `WAVEMAX` is 'Maximum observed (vacuum) wavelength (Angstroms)'"

# ╔═╡ 3c9d8afd-e1e9-4cb4-8f5d-64dab93168e4
begin
	specobj_fits = FITS("data/specObj-dr12.fits")
		specobj = DataFrame(
			plate=read(specobj_fits[2], "PLATE"),
			mjd=read(specobj_fits[2], "MJD"),
			fiberid=read(specobj_fits[2], "FIBERID"),
			wavemin=read(specobj_fits[2], "WAVEMIN"),
			wavemax=read(specobj_fits[2], "WAVEMAX"))
end

# ╔═╡ cc76dfe5-9e7c-49ef-ab72-97afed20ccb7
wave_subset = leftjoin(subset, specobj, on=[:plate, :mjd, :fiberid])

# ╔═╡ b319b899-622b-45ba-9451-b938a8f09ced
begin
	wavemin_zero_idx = wave_subset[:wavemin] .== 0.0
	wavemax_zero_idx = wave_subset[:wavemax] .== 0.0
	sum(wavemin_zero_idx), sum(wavemax_zero_idx)
end

# ╔═╡ 900fd965-17ba-4a28-9e63-ca3dbc03975b
begin
	wavemin = quantile(wave_subset[.~wavemin_zero_idx, :wavemin], 0.99)
	wavemax = quantile(wave_subset[.~wavemax_zero_idx, :wavemax], 0.01)
	logwavemin, logwavemax = log10(wavemin), log10(wavemax)
	wavemin, wavemax, logwavemin, logwavemax
end

# ╔═╡ 36e49ea5-41a7-4034-ad58-1354288e0cf0
begin
	wave_idx = (wave_subset[:wavemin] .<= wavemin) .& (wavemax .<= wave_subset[:wavemax])
	sum(wave_idx)
end

# ╔═╡ 7e4db827-36f9-4aa0-a7a2-c213e6cd8126
final_subset = wave_subset[wave_idx, :]

# ╔═╡ 5d6d6774-0587-4649-a230-24a2519e81f0
md"## HDF5"

# ╔═╡ d6165f9c-41ee-4b3e-b7fb-004e3e4c4825
begin
	id = Matrix{Int32}(undef, 3, size(final_subset, 1))
	id[1, :] = final_subset[:plate]
	id[2, :] = final_subset[:mjd]
	id[3, :] = final_subset[:fiberid]
	id
end

# ╔═╡ 49ac94ac-1563-4516-97b5-744ea9d55f93
begin
	# read-write, create file if not existing, preserve existing contents
	fid = h5open("data/dr12q_superset.hdf5", "cw")
	write_dataset(fid, "id", id)
	write_dataset(fid, "z_vi", convert(Vector{Float32}, final_subset[:z_vi]))
	write_dataset(fid, "z_pipe", convert(Vector{Float32}, final_subset[:z_pipe]))
	close(fid)
end

# ╔═╡ Cell order:
# ╟─24ee40d4-7e63-4654-b112-b7ecb7b912d2
# ╠═2a2861e4-9d00-11eb-08fd-972e78fd9d18
# ╠═6779d063-2ad5-4a09-9bb8-22cb97ba711d
# ╠═43c3d594-e3a1-4af1-b093-f32fcd5b7e70
# ╟─d93bd116-4d8e-42cb-a27c-9a7ee06d67dd
# ╠═eb734e4c-0a9c-43c0-a9ff-fe1fab245783
# ╟─0bfe5e2b-8095-412d-85f3-aa373ab20cc9
# ╠═541e307d-daa7-492c-95b4-6f835252d4d8
# ╟─7ffb928d-cdff-4eb6-aaf0-86a132f23572
# ╠═43f3be7c-aada-4e66-aee8-33abecabe7e8
# ╟─83889e5c-e5d3-4afb-a726-c5ee7c7e7969
# ╠═3c9d8afd-e1e9-4cb4-8f5d-64dab93168e4
# ╠═cc76dfe5-9e7c-49ef-ab72-97afed20ccb7
# ╠═b319b899-622b-45ba-9451-b938a8f09ced
# ╠═900fd965-17ba-4a28-9e63-ca3dbc03975b
# ╠═36e49ea5-41a7-4034-ad58-1354288e0cf0
# ╠═7e4db827-36f9-4aa0-a7a2-c213e6cd8126
# ╟─5d6d6774-0587-4649-a230-24a2519e81f0
# ╠═d6165f9c-41ee-4b3e-b7fb-004e3e4c4825
# ╠═49ac94ac-1563-4516-97b5-744ea9d55f93
