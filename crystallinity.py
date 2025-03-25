import arsenal as ars
import utils
from ovito import scene
from ovito.io import import_file
from ovito.modifiers import ComputePropertyModifier
from ovito.modifiers import ColorCodingModifier
from ovito.vis import Viewport, SimulationCellVis
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pickle
import os

## Description
# this script reads either a caDNAno file or oxDNA configuration file to
  # get the positions for visualizing a DNA origami design, then it writes
  # the lammps-style geometry file for the visualization in OVITO.
# this script will only work if "backend_basics.py" has already been run for
  # the given simulation (requires a populated "analysis" folder).

## To Do
# correlate with individual staple binding times, moving average


################################################################################
### Parameters

def main():

	### input files
	simID = "16HB"
	simTag = ""
	srcFold = "/Users/dduke/Files/dnafold_lmp/"
	multiSim = False

	### analysis options
	position_src = "oxdna"			# where to get bead locations (cadnano or oxdna)
	avgCorrStrand = True			# whether to average SHybCorr over strand for visual

	### results source options
	loadResults = False
	simHomeFold = srcFold + simID + simTag + "/"
	resultsFile = simHomeFold + "analysis/crystallinity_results.pkl"

	### read data and calculate results
	if not loadResults:

		### data reading parameters
		nstep_skip = 0  				# number of recorded initial steps to skip
		coarse_time = 1 				# coarse factor for time steps

		### get simulation folders
		simFolds, nsim = utils.getSimFolds(simHomeFold, multiSim)

		### get pickled data
		connFile = simHomeFold + "analysis/connectivity_vars.pkl"
		ars.testFileExist(connFile,"connectivity")
		with open(connFile,"rb") as f:
			[strands, bonds_backbone, complements, n_scaf] = pickle.load(f)[:4]

		### loop through simulations
		S_allSim = nsim*[None]
		first_hyb_times_scaled_allSim = np.zeros((nsim,n_scaf))
		for i in range(nsim):

			### calculate crystallinity
			datFile = simFolds[i] + "analysis/trajectory_centered_scaf.dat"
			points, _, dbox = ars.readAtomDump(datFile, nstep_skip, coarse_time)
			dump_every_S = ars.getDumpEvery(datFile)*coarse_time
			S_allSim[i] = calcCrystallinity(points, dbox)

			### calculate hyb times
			hybFile = simFolds[i] + "analysis/hyb_status.dat"
			hyb_status, dump_every_hyb = utils.readHybStatus(hybFile)
			first_hyb_times_scaled_allSim[i,:] = utils.calcFirstHybTimes(hyb_status, complements, n_scaf, dump_every_hyb)[1]

		### store results
		with open(resultsFile, 'wb') as f:
			pickle.dump([S_allSim, first_hyb_times_scaled_allSim, strands, bonds_backbone, complements, n_scaf, dump_every_S], f)

	### load results
	else:
		ars.testFileExist(resultsFile,"results")
		with open(resultsFile, 'rb') as f:
			[S_allSim, first_hyb_times_scaled_allSim, strands, bonds_backbone, complements, n_scaf, dump_every_S] = pickle.load(f)


################################################################################
### Results

	### average crystallinity data
	S_allSim, S_avg, S_sem, nstep_min = averageCrystallinityData(S_allSim)
	Sfinal_allSim = S_allSim[:,-1]

	### multiple simulation analysis
	if multiSim:

		### report best simulations
		copyNames = ars.readCopies(simHomeFold + "copies.txt")[0]
		sorted_results = sorted(zip(Sfinal_allSim,copyNames), reverse=True)
		for Sfinal, copyName in sorted_results:
			print(f"{copyName}: {Sfinal:0.4f}")


		### crystallinity / hyb times correlation visual
		SHybCorr, SHybCorr_strand = calcSHybCorr(first_hyb_times_scaled_allSim, Sfinal_allSim, strands, complements)
		r = utils.getIdealPositions(simID, simHomeFold, position_src)
		r, charges, bonds, dbox3 = prepGeoData(r, strands, bonds_backbone, complements, SHybCorr, SHybCorr_strand, avgCorrStrand)

		### write geometry and ovito files
		outGeoFile = srcFold + simID + simTag + "/analysis/geometry_SHybCorr.in"
		ars.writeGeo(outGeoFile, dbox3, r, types=strands, charges=charges, bonds=bonds)
		ovitoFile = srcFold + simID + simTag + "/analysis/vis_SHybCorr.ovito"
		writeOvito(ovitoFile, outGeoFile)

		### report best staples
		if avgCorrStrand:
			sorted_results = sorted(zip(SHybCorr_strand,np.arange(len(SHybCorr_strand))+1), reverse=True)
			for corr, s in sorted_results:
				print(f"{s:2.0f}: {corr:0.4f}")

	### plot
	plotCrystallinity(S_allSim, S_avg, S_sem, dump_every_S)
	plt.show()



################################################################################
### Plotters

### plot average Landau-De Gennes crystallinity parameter
def plotCrystallinity(Ss, S_avg, S_sem, dump_every):

	### calculate time
	dt = 0.01
	scale = 5200
	nstep_min = len(S_avg)
	time = np.arange(nstep_min)*dump_every*dt*scale*1E-9

	### plot prep
	cmap = cm.winter
	norm = mcolors.Normalize(vmin=min(Ss[:,-1]), vmax=max(Ss[:,-1]))
	nsim = len(Ss)

	### plot
	ars.magicPlot()
	plt.figure("S",figsize=(8,6))
	if nsim > 1:
		for i in range(nsim):
			color = cmap(norm(Ss[i,-1]))
			plt.plot(time,Ss[i],color=color,alpha=0.4)
		plt.fill_between(time,S_avg-S_sem,S_avg+S_sem,color='k',alpha=0.4,edgecolor='none')
	plt.plot(time,S_avg,'k',linewidth=2)
	plt.xlabel("Simulation Progression [$s$]")
	plt.ylabel("$S$")
	plt.title("Landau-De Gennes Crystallinity")


### write session state vito file that visualizes the geometry
def writeOvito(ovitoFile, outGeoFile):

	### get base geometry
	pipeline = import_file(outGeoFile, atom_style="full")
	pipeline.add_to_scene()

	### disable simulation cell
	vis_element = pipeline.source.data.cell.vis
	vis_element.enabled = False

	### set active viewport to top perspective
	viewport = scene.viewports.active_vp
	viewport.type = Viewport.Type.PERSPECTIVE
	viewport.camera_dir = (-1,0,0)
	viewport.camera_up = (0,1,0)
	viewport.zoom_all()

	### add modifiers
	pipeline.modifiers.append(ComputePropertyModifier(output_property='Radius',expressions=['(ParticleType==1)?0.6:1']))
	pipeline.modifiers.append(ComputePropertyModifier(operate_on='bonds',output_property='Width',expressions=['(BondType==1)?1.2:2']))
	pipeline.modifiers.append(ColorCodingModifier(property='Charge',start_value=-1,end_value=1,gradient=ColorCodingModifier.BlueWhiteRed()))

	### write ovito file
	scene.save(ovitoFile)


################################################################################
### Calculation Managers

def calcCrystallinity(points, dbox):
	nstep = points.shape[0]
	n_scaf = points.shape[1]
	S = np.zeros(nstep)
	for i in range(nstep):
		dX = np.zeros((n_scaf-1,3))
		for j in range(n_scaf-1):
			dX[j] = ars.applyPBC( points[i,j+1] - points[i,j], dbox )
		Q_val = np.zeros((n_scaf-1,3,3))
		for j in range(n_scaf-1):
			dX_mag = np.linalg.norm(dX[j])
			if dX_mag != 0:
				for m in range(3):
					for n in range(3):
						v1 = dX[j,m] / dX_mag
						v2 = dX[j,n] / dX_mag
						Q_val[j,m,n] = 3/2*v1*v2 - 1/2*(i==j)
		Q = np.mean(Q_val, axis=0)
		S[i] = np.max(np.linalg.eigvals(Q).real) - 1/2
	return S


### average the crystallinity across several simulations
def averageCrystallinityData(Ss):
	nsim = len(Ss)
	nstep_min = min([len(i) for i in Ss])
	Ss_trim = np.zeros((nsim,nstep_min))
	for i in range(nsim):
		Ss_trim[i] = Ss[i][:nstep_min]
	S_avg = np.mean(Ss_trim, axis=0)
	S_sem = np.zeros(nstep_min)
	for i in range(nstep_min):
		S_sem[i] = ars.calcSEM(Ss_trim[:,i])
	return Ss_trim, S_avg, S_sem, nstep_min


### calculate correlation between crystallinity and hybridization times
def calcSHybCorr(first_hyb_times_scaled_allSim, Sfinal_allSim, strands, complements):
	n_scaf = first_hyb_times_scaled_allSim.shape[1]
	SHybCorr = np.zeros(n_scaf)
	for i in range(n_scaf):
		if not (first_hyb_times_scaled_allSim[:,i]==0).all():
			SHybCorr[i] = np.corrcoef(first_hyb_times_scaled_allSim[:,i],Sfinal_allSim)[0,1]

	### average over strands
	nstrand = max(strands)
	SHybCorr_strand = np.zeros(nstrand)
	SHybCorr_strand[0] = np.mean(SHybCorr[SHybCorr!=0])
	ncomp_strand = np.zeros(nstrand)
	ncomp_strand[0] = 1
	for i in range(n_scaf):
		if len(complements[i]) > 0:
			strand = strands[complements[i][0]-1]
			ncomp_strand[strand-1] += 1
			SHybCorr_strand[strand-1] += SHybCorr[i]
	SHybCorr_strand *= 1/ncomp_strand

	### results
	return SHybCorr, SHybCorr_strand


################################################################################
### Utility Functions

### get geometry data ready for visualization
def prepGeoData(r, strands, bonds, complements, SHybCorr, SHybCorr_strand, avgCorrStrand):
	n_ori = len(strands)
	n_scaf = np.sum(strands==1)

	### box diameter
	dbox3 = [ max(abs(r[:,0]))+2.72, max(abs(r[:,1]))+2.4, max(abs(r[:,2]))+2.4 ]
	dbox3 = [ 2*i for i in dbox3 ]

	### assign charge
	charges = np.zeros(n_ori)
	charges[:n_scaf] = 0
	for i in range(n_scaf,n_ori):
		if avgCorrStrand:
			charges[i] = SHybCorr_strand[strands[i]-1]
		else:
			charges[i] = SHybCorr[complements[i][0]-1]

	### change bond types
	bonds[bonds[:,1] > n_scaf, 0] = 2

	### return results
	return r, charges, bonds, dbox3


### run the script
if __name__ == "__main__":
	main()

