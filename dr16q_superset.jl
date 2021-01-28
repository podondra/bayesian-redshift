### A Pluto.jl notebook ###
# v0.12.19

using Markdown
using InteractiveUtils

# ╔═╡ 3621f840-5fcd-11eb-3be8-15ac4ab1b566
using DataFrames, DelimitedFiles, FITSIO, HDF5, Printf, Statistics, StatsPlots

# ╔═╡ 05685b0c-5fcf-11eb-1f7f-c77372b9790e
md"# DR16Q Superset Catalogue

> The superset contains 1,440,615 observations of quasars, stars, and galaxies that were all targeted as quasars (or appeared in previous quasar catalogs)."

# ╔═╡ 5e927e1c-5fcd-11eb-39cd-d19c696d924e
begin
	superset_fits = FITS("data/DR16Q_Superset_v3.fits")
	superset = DataFrame(
		plate=read(superset_fits[2], "PLATE"),
		mjd=read(superset_fits[2], "MJD"),
		fiberid=read(superset_fits[2], "FIBERID"),
		z_qn=read(superset_fits[2], "Z_QN"),
		random_select=read(superset_fits[2], "RANDOM_SELECT"),
		z_10k=read(superset_fits[2], "Z_10K"),
		z_conf_10k=read(superset_fits[2], "Z_CONF_10K"),
		pipe_corr_10k=read(superset_fits[2], "PIPE_CORR_10K"),
		z_vi=read(superset_fits[2], "Z_VI"),
		z_conf=read(superset_fits[2], "Z_CONF"),
		class_person=read(superset_fits[2], "CLASS_PERSON"),
		z=read(superset_fits[2], "Z"),
		source_z=read(superset_fits[2], "SOURCE_Z"),
		z_pip=read(superset_fits[2], "Z_PIPE"),
		zwarning=read(superset_fits[2], "ZWARNING"),
		z_pca=read(superset_fits[2], "Z_PCA"),
		zwarn_pca=read(superset_fits[2], "ZWARN_PCA"))
end

# ╔═╡ 9e7bd468-5fcf-11eb-24d2-a349a471fc8e
md"> For objects that have a redshift in the columns Z\_VI or Z\_10K and a confidence (Z\_CONF or Z\_CONF\_10K) of ≥ 2, Z records the corresponding redshift and SOURCE\_Z is set to VI.
> Otherwise, if an object has a redshift in the columns Z\_DR6Q\_HW or Z\_DR7Q\_SCH these values are used (with Z\_DR6Q\_HW overriding Z\_DR7Q\_SCH) and SOURCE\_Z is set to DR6Q\_HW or DR7QV\_SCH. As the ZDR7Q\_HW redshifts did not formally appear in the Shen et al. (2011) paper, these values are not used to populate the Z column.
> If no other visual inspection redshift is populated then Z\_DR12Q is used (and SOURCE\_Z is set to DR12QV).
> For objects with DR12Q redshifts, only the visual inspection redshifts are recorded; DR12Q pipeline redshifts  are not included.
> In the absence of any of these visual inspection redshifts, Z is populated with the automated pipeline redshift (and SOURCE\_Z is set to PIPE)."

# ╔═╡ c82279c0-5fcf-11eb-17b9-1136e2a2792d
begin
	source_idx = superset[:source_z] .!= "PIPE"
	gt_zero_idx = superset[:z] .> 0
	# TODO should I include epsilon?
	eq_zero_idx = superset[:z] .== 0
	sum(source_idx), sum(gt_zero_idx), sum(eq_zero_idx)
end

# ╔═╡ 50475460-5fd0-11eb-0ec7-7301438e4cd2
source_subset = superset[source_idx .& (gt_zero_idx .| eq_zero_idx), :]

# ╔═╡ 282e7e62-5fd1-11eb-27ba-7d4d50fc1374
@df source_subset density(:z, xlabel="z", ylabel="Density", legend=:none)

# ╔═╡ 868625f0-5fd1-11eb-00dd-1f68c388ceb6
md"## Wavelength Range"

# ╔═╡ 86d7f01c-613c-11eb-111f-1d47ffe5dfd3
begin
	specobj_fits = FITS("data/specObj-dr16.fits")
	specobj = DataFrame(
		plate=read(specobj_fits[2], "PLATE"),
		mjd=read(specobj_fits[2], "MJD"),
		fiberid=read(specobj_fits[2], "FIBERID"),
		wavemin=read(specobj_fits[2], "WAVEMIN"),
		wavemax=read(specobj_fits[2], "WAVEMAX"),
		wcoverage=read(specobj_fits[2], "WCOVERAGE"))
end

# ╔═╡ 6cb7f21c-5fd2-11eb-324c-6755e1ad21db
wave_subset = leftjoin(source_subset, specobj, on=[:plate, :mjd, :fiberid])

# ╔═╡ ebad79d8-6148-11eb-20c6-e556e1e41fa3
dropmissing!(wave_subset, [:wavemin, :wavemax])

# ╔═╡ 62db6500-6145-11eb-16e4-af95d069068e
describe(wave_subset[[:wavemin, :wavemax]])

# ╔═╡ f9b6140a-6147-11eb-2b7c-4bdad13b9ace
begin
	wavemin = quantile(wave_subset[:wavemin], 0.95)
	wavemax = quantile(wave_subset[:wavemax], 0.05)
	logwavemin, logwavemax = log10(wavemin), log10(wavemax)
	wavemin, wavemax, logwavemin, logwavemax
end

# ╔═╡ 6f90500a-6148-11eb-2645-61e9174bf9bd
begin
	wave_idx = (wave_subset[:wavemin] .< wavemin) .& (wave_subset[:wavemax] .> wavemax)
	sum(wave_idx)
end

# ╔═╡ c8917b70-6148-11eb-038a-5d388d41b053
final_subset = wave_subset[wave_idx, :]

# ╔═╡ 2920ca40-6149-11eb-0fd5-3769db3f7c26
@df final_subset density(:wcoverage, xlabel="wcoverage", ylabel="Density", legend=:none)

# ╔═╡ 6a2519f6-6149-11eb-2724-d14f7f0cc7a5
md"## Filepaths"

# ╔═╡ 864db836-6149-11eb-1565-ab1718ca86f1
function fits_filepath(plate, mjd, fiberid)
	@sprintf("%04d/spec-%04d-%05d-%04d.fits", plate, plate, mjd, fiberid)
end

# ╔═╡ 448746a8-614a-11eb-2854-13af398dca65
filepaths = fits_filepath.(
	final_subset[:plate], final_subset[:mjd], final_subset[:fiberid])

# ╔═╡ d3830dee-614a-11eb-1fc8-2bce6cd456f0
writedlm("data/spec.lst", filepaths)

# ╔═╡ 51962842-614b-11eb-20a7-434c8d7421ea
md"## Preview"

# ╔═╡ 5a0160dc-614b-11eb-261c-5f67462a042d
begin
	filepath = "data/DR16Q_Superset_v3/" * rand(filepaths)
	hdul = FITS(filepath)
	flux = read(hdul[2], "flux")
	loglam = read(hdul[2], "loglam")
	lam = 10 .^ loglam
	l = @layout [a; b]
	p1 = plot(lam, flux)
	idx = (logwavemin .<= loglam) .& (loglam .<= logwavemax)
	p2 = plot(lam[idx], flux[idx])
	plot(p1, p2, layout=l)
end

# ╔═╡ c66ea292-614c-11eb-2bc7-675d0b185c03
md"## HDF5"

# ╔═╡ d89f83d6-614d-11eb-170b-e1a7febab65e
begin
	# read-write, create file if not existing, preserve existing contents
	fid = h5open("data/dr16q_superset.hdf5", "cw")
	id = Matrix(final_subset[[:plate, :mjd, :fiberid]])
	write_dataset(fid, "id", id)
	write_dataset(fid, "z", final_subset[:z])
	close(fid)
end

# ╔═╡ Cell order:
# ╟─05685b0c-5fcf-11eb-1f7f-c77372b9790e
# ╠═3621f840-5fcd-11eb-3be8-15ac4ab1b566
# ╠═5e927e1c-5fcd-11eb-39cd-d19c696d924e
# ╟─9e7bd468-5fcf-11eb-24d2-a349a471fc8e
# ╠═c82279c0-5fcf-11eb-17b9-1136e2a2792d
# ╠═50475460-5fd0-11eb-0ec7-7301438e4cd2
# ╠═282e7e62-5fd1-11eb-27ba-7d4d50fc1374
# ╟─868625f0-5fd1-11eb-00dd-1f68c388ceb6
# ╠═86d7f01c-613c-11eb-111f-1d47ffe5dfd3
# ╠═6cb7f21c-5fd2-11eb-324c-6755e1ad21db
# ╠═ebad79d8-6148-11eb-20c6-e556e1e41fa3
# ╠═62db6500-6145-11eb-16e4-af95d069068e
# ╠═f9b6140a-6147-11eb-2b7c-4bdad13b9ace
# ╠═6f90500a-6148-11eb-2645-61e9174bf9bd
# ╠═c8917b70-6148-11eb-038a-5d388d41b053
# ╠═2920ca40-6149-11eb-0fd5-3769db3f7c26
# ╟─6a2519f6-6149-11eb-2724-d14f7f0cc7a5
# ╠═864db836-6149-11eb-1565-ab1718ca86f1
# ╠═448746a8-614a-11eb-2854-13af398dca65
# ╠═d3830dee-614a-11eb-1fc8-2bce6cd456f0
# ╟─51962842-614b-11eb-20a7-434c8d7421ea
# ╠═5a0160dc-614b-11eb-261c-5f67462a042d
# ╟─c66ea292-614c-11eb-2bc7-675d0b185c03
# ╠═d89f83d6-614d-11eb-170b-e1a7febab65e
