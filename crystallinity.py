import armament as ars
import utils
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pickle
import sys
import os

## Description
# this script reads a DNAfold trajectory (or a batch of trajectories) and
  # calculates the crystallinity of the scaffold throughout the simulation.
# this script will only work if "backend_basics.py" has already been run for
  # the given simulation (requires a populated "analysis" folder).


################################################################################
### Parameters

def main():

	### get arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--copiesFile',		type=str,	default=None,		help='name of copies file, which contains a list of simulation folders')	
	parser.add_argument('--simFold',		type=str,	default=None,		help='name of simulation folder, should exist within current directory')
	parser.add_argument('--rseed',			type=int,	default=1,			help='random seed, used to find simFold if necessary')
	parser.add_argument('--clusterFile',	type=str,	default=None,		help='name of cluster file, ')	
	parser.add_argument('--loadResults',	type=bool,	default=False,		help='whether to load the results from a pickle file')
	parser.add_argument('--nstep_skip',		type=int,	default=0,			help='if not loading results, number of recorded initial steps to skip')
	parser.add_argument('--coarse_time',	type=int,	default=1,			help='if not loading results, coarse factor for time steps')
	parser.add_argument('--mov_avg_stride',	type=int,	default=1,			help='if not loading results, stride length for moving average')

	### set arguments
	args = parser.parse_args()
	copiesFile = args.copiesFile
	simFold = args.simFold
	rseed = args.rseed
	clusterFile = args.clusterFile
	loadResults = args.loadResults
	nstep_skip = args.nstep_skip
	coarse_time = args.coarse_time
	mov_avg_stride = args.mov_avg_stride

	### check arguments
	if copiesFile is not None and clusterFile is not None:
		clusterFile = None
		print("Flag: Both copies and cluster file provided, so ignoring cluster file.")


################################################################################
### Heart

	### read data and calculate results
	if not loadResults:

		### get simulation folders
		simFolds, nsim = utils.getSimFolds(copiesFile, simFold, rseed)

		### get pickled data
		connFile = "analysis/connectivity_vars.pkl"
		ars.testFileExist(connFile, "connectivity")
		with open(connFile, 'rb') as f:
			n_scaf = pickle.load(f)[3]

		### get minimum number of steps
		nstep_allSim = np.zeros(nsim,dtype=int)
		for i in range(nsim):
			datFile = simFolds[i] + "analysis/trajectory_centered.dat"
			nstep_allSim[i] = ars.getNstep(datFile, nstep_skip, coarse_time)
		nstep_min = int(min(nstep_allSim))

		### get trajectory dump frequency
		datFile = simFolds[0] + "analysis/trajectory_centered.dat"
		dump_every = ars.getDumpEvery(datFile)*coarse_time

		### whole scaffold analysis
		if clusterFile is None:
			bdis = list(range(1,n_scaf+1))

			### loop through simulations
			S_allSim = np.zeros((nsim,nstep_min))
			for i in range(nsim):

				### calculate crustallinity
				datFile = simFolds[i] + "analysis/trajectory_centered.dat"
				points, _, dbox = ars.readAtomDump(datFile, nstep_skip, coarse_time, bdis=bdis, nstep_max=nstep_min); print("")
				S = utils.calcCrystallinity(points, dbox)
				S_allSim[i] = ars.movingAvg(S, mov_avg_stride)

		### clustered scaffold analysis
		else:
			bdis = ars.readCluster(clusterFile)
			ncluster = len(bdis)

			### read trajectory
			datFile = simFolds[0] + "analysis/trajectory_centered.dat"
			points, _, dbox, molecules = ars.readAtomDump(datFile, nstep_skip, coarse_time, bdis=bdis, nstep_max=nstep_min); print("")
			points = ars.sortPointsByMolecule(points, molecules)

			### loop through clusters
			S_allSim = np.zeros((ncluster,nstep_min))
			for i in range(ncluster):

				### calculate crustallinity
				S = utils.calcCrystallinity(points[i], dbox)
				S_allSim[i] = ars.movingAvg(S, mov_avg_stride)

		### store results
		resultsFile = "analysis/crystallinity_vars.pkl"
		with open(resultsFile, 'wb') as f:
			pickle.dump([S_allSim, dump_every], f)

	### load results
	else:
		resultsFile = "analysis/crystallinity_vars.pkl"
		ars.testFileExist(resultsFile,"results")
		with open(resultsFile, 'rb') as f:
			[S_allSim, dump_every] = pickle.load(f)


################################################################################
### Results

	### report best simulations
	if copiesFile is not None:
		copyNames = ars.readCopies(copiesFile)[0]
		sorted_results = sorted(zip(S_allSim[:,-1],copyNames), reverse=True)
		for S_final, copyName in sorted_results:
			print(f"{copyName}: {S_final:0.4f}")

	### plot
	plotCrystallinity(S_allSim, dump_every)
	plt.show()


################################################################################
### Plotters

### plot average Landau-De Gennes crystallinity parameter
def plotCrystallinity(S_allSim, dump_every):
	nsim = S_allSim.shape[0]
	nstep = S_allSim.shape[1]

	### calculations
	S_avg = np.mean(S_allSim, axis=0)
	S_sem = np.zeros(nstep)
	for i in range(nstep):
		S_sem[i] = ars.calcSEM(S_allSim[:,i])

	### calculate time
	dt = 0.01
	scale = 5200
	time = np.arange(nstep)*dump_every*dt*scale*1E-9

	### plot prep
	cmap = cm.winter
	norm = mcolors.Normalize(vmin=min(S_allSim[:,-1]), vmax=max(S_allSim[:,-1]))

	### plot
	ars.magicPlot()
	plt.figure("S",figsize=(8,6))
	if nsim > 1:
		for i in range(nsim):
			color = cmap(norm(S_allSim[i,-1]))
			plt.plot(time,S_allSim[i],color=color,alpha=0.4)
		plt.fill_between(time,S_avg-S_sem,S_avg+S_sem,color='k',alpha=0.4,edgecolor='none')
	plt.plot(time,S_avg,'k',linewidth=2)
	plt.xlabel("Time [$s$]")
	plt.ylabel("$S$")
	plt.title("Landau-De Gennes Crystallinity")


### run the script
if __name__ == "__main__":
	main()

