import arsenal as ars
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

## To Do
# correlate with individual staple binding times, moving average


################################################################################
### Parameters

def main():

	### input files
	simID = "16HB"
	simTag = "/resNo"
	srcFold = "/Users/dduke/Files/dnafold_lmp/production/"
	multiSim = True

	### results source options
	loadResults = False
	resultsFold = srcFold + simID + simTag + "/analysis/"
	resultsFile = resultsFold + "crystallinity_results.pkl"

	### read data and calculate results
	if not loadResults:

		### data reading parameters
		nstep_skip = 0  				# number of recorded initial steps to skip
		coarse_time = 100 				# coarse factor for time steps

		### simgle simulation analysis
		if not multiSim:
			nsim = 1
			copyNames = [None]
			simFolds = [ srcFold + simID + simTag + "/" ]

		### multiple simulation folders
		else:
			copiesFold = srcFold + simID + simTag + "/"
			copiesFile = copiesFold + "copies.txt"
			copyNames, nsim = ars.readCopies(copiesFile)
			simFolds = [ copiesFold + copyNames[i] + "/" for i in range(nsim) ]

		### loop through simulations
		Ss = [None]*nsim
		for i in range(nsim):

			### calculate crystallinity
			datFile = simFolds[i] + "analysis/trajectory_centered_scaf.dat"
			points, _, dbox = ars.readAtomDump(datFile, nstep_skip, coarse_time)
			dump_every = ars.getDumpEvery(datFile)*coarse_time
			Ss[i] = calcCrystallinity(points, dbox)

		### store results
		ars.createSafeFold(resultsFold)
		with open(resultsFile, 'wb') as f:
			pickle.dump([Ss, copyNames, dump_every], f)

	### load results
	else:
		ars.testFileExist(resultsFile,"results")
		with open(resultsFile, 'rb') as f:
			[Ss, copyNames, dump_every] = pickle.load(f)


################################################################################
### Results
	
	### average the data
	Ss, S_avg, S_sem, nstep_min = averageCrystallinityData(Ss)

	### report
	if len(copyNames) > 1:
		sorted_results = sorted(zip(Ss[:,-1],copyNames), reverse=True)
		for S, copyName in sorted_results:
			print(f"{copyName}: {S:0.4f}")

	### plot
	plotCrystallinity(Ss, S_avg, S_sem, dump_every)
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
	nsim = len(Ss)
	ars.magicPlot()

	### plot
	cmap = cm.winter
	norm = mcolors.Normalize(vmin=min(Ss[:,-1]), vmax=max(Ss[:,-1]))
	plt.figure("S",figsize=(8,6))
	for i in range(nsim):
		color = cmap(norm(Ss[i,-1]))
		plt.plot(time,Ss[i],color=color,alpha=0.4)
	plt.plot(time,S_avg,'k',linewidth=2)
	plt.fill_between(time,S_avg-S_sem,S_avg+S_sem,color='k',alpha=0.4,edgecolor='none')
	plt.xlabel("Simulation Progression [$s$]")
	plt.ylabel("$S$")
	plt.title("Landau-De Gennes Crystallinity")


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


### run the script
if __name__ == "__main__":
	main()

