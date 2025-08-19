import armament as ars
import utils
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

## Description
# this script reads a batch of hybridization time trajectories, calculates
  # the free staple kinetics and number of hybridizations, and plots them.
# this script will only work if "backend_basics.py" has already been run for
  # the given simulation (requires a populated "analysis" folder).


################################################################################
### Parameters

def main():

	### get arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--copiesFile',		type=str,	required=True,	help='name of copies file, which contains a list of simulation folders')	
	parser.add_argument('--nstep_skip',		type=float,	default=0,		help='number of recorded initial steps to skip')
	parser.add_argument('--coarse_time',	type=int,	default=1,		help='coarse factor for time steps')
	parser.add_argument('--nstep_max',		type=float,	default=0,		help='max number of recorded steps to use (0 for all)')
	parser.add_argument('--misFile',		type=str,	default=None,	help='name of misbinding file, which contains cutoffs and energies')
	parser.add_argument('--saveFig',		type=int,	default=False,	help='whether to save pdf of figures')

	### set arguments
	args = parser.parse_args()
	copiesFile = args.copiesFile
	nstep_skip = int(args.nstep_skip)
	coarse_time = args.coarse_time
	nstep_max = int(args.nstep_max)
	misFile = args.misFile
	saveFig = args.saveFig

	### get simulation folders
	simFolds, nsim = utils.getSimFolds(copiesFile)


################################################################################
### Heart

	### get pickled data
	connFile = "analysis/connectivity_vars.pkl"
	strands, n_scaf, nbead = readConn(connFile)

	### get minimum number of steps
	nstep_allSim = np.zeros(nsim,dtype=int)
	for i in range(nsim):
		datFile = simFolds[i] + "analysis/trajectory_centered.dat"
		nstep_allSim[i] = ars.getNstep(datFile, nstep_skip, coarse_time)
	nstep_min = int(min(nstep_allSim))
	nstep_use = nstep_min if nstep_max == 0 else min([nstep_min,nstep_max])

	### get hybridization dump frequency
	hybFile = simFolds[0] + "analysis/hyb_status.dat"
	dump_every = utils.getDumpEveryHyb(hybFile)

	### loop over simulations
	hyb_status_allSim = np.zeros((nsim,nstep_use,nbead))
	for i in range(nsim):

		### analyze hybridizations
		hybFile = simFolds[i] + "analysis/hyb_status.dat"; print()
		hyb_status_allSim[i] = utils.readHybStatus(hybFile, nstep_skip, coarse_time, nstep_use, 'mis')

	### create figures folder
	if saveFig: ars.createSafeFold("analysis/figures")

	### plot native complement kinetics
	plotKinetics(hyb_status_allSim, strands, n_scaf, dump_every)
	if saveFig: plt.savefig("analysis/figures/kinetics.pdf")

	### plot number of native complement hybridizations
	plotNhyb(hyb_status_allSim, n_scaf, dump_every)
	if saveFig: plt.savefig("analysis/figures/hyb_count.pdf")

	### misbinding analysis
	if misFile is not None:
		### plot native complement kinetics
		plotKinetics(hyb_status_allSim, strands, n_scaf, dump_every, True)
		if saveFig: plt.savefig("analysis/figures/kinetics_mis.pdf")

		### plot total number of hybridizations
		plotNhyb(hyb_status_allSim, n_scaf, dump_every, True)
		if saveFig: plt.savefig("analysis/figures/hyb_count_mis.pdf")

		### plot number of misbound hybridizations
		plotNmis(hyb_status_allSim, n_scaf, dump_every, misFile)
		if saveFig: plt.savefig("analysis/figures/mis_count.pdf")

	### display
	if not saveFig: plt.show()


################################################################################
### Plotters

### plot free staple kinetics
def plotKinetics(hyb_status_allSim, strands, n_scaf, dump_every, includeMis=False):
	figLabel = "Kinetics"

	### adjustments for including misbinding
	if includeMis:
		hyb_status_allSim[hyb_status_allSim>1] = 1
		figLabel = "Mis Kinetics"

	### calculate free staple concentrations
	nsim = hyb_status_allSim.shape[0]
	nstep = hyb_status_allSim.shape[1]
	nbead = hyb_status_allSim.shape[2]
	nstap = max(strands)-1
	conc_min = 1/nstap
	conc_avg = np.zeros(nstep)
	conc_sem = np.zeros(nstep)
	for i in range(nstep):
		stap_freedom = np.ones((nstap,nsim),dtype=int)
		for j in range(n_scaf,nbead):
			for k in range(nsim):
				if hyb_status_allSim[k,i,j] == 1:
					stap_freedom[strands[j]-2,k] = 0
		conc_indiv = np.mean(stap_freedom,axis=1)
		conc_avg[i] = max(np.mean(conc_indiv), conc_min)
		conc_sem[i] = ars.calcSEM(conc_indiv)
	time = utils.getTime(nstep, dump_every)

	### calculate uncertainty
	conc_min_arr = conc_min*np.ones(nstep)
	error_min = np.maximum(np.log(conc_avg-conc_sem),np.log(conc_min_arr))
	error_max = np.log(conc_avg+conc_sem)

	### plot
	ars.magicPlot()
	plt.figure(figLabel)
	plt.plot(time, np.log(conc_avg))
	plt.fill_between(time, error_min, error_max, alpha=0.3)
	plt.xlabel("Time [s]")
	plt.ylabel("$\\ln(C/C_0)$")
	plt.ylim(np.log(conc_min)*1.05, -np.log(conc_min)*0.05)
	plt.title("Free Staple Kinetics")
	plt.legend(['Mean','SEM'])


### plot number of hybridizations
def plotNhyb(hyb_status_allSim, n_scaf, dump_every, includeMis=False):
	figLabel = "N Hyb"

	### adjustments for including misbinding
	if includeMis:
		hyb_status_allSim[hyb_status_allSim>1] = 1
		figLabel = "Mis N Hyb"

	### calculate free staple concentrations
	nsim = hyb_status_allSim.shape[0]
	nstep = hyb_status_allSim.shape[1]
	n_hyb_avg = np.zeros(nstep)
	n_hyb_sem = np.zeros(nstep)
	for i in range(nstep):
		n_hyb_indiv = np.sum(hyb_status_allSim[:,i,:n_scaf]==1,axis=1)
		n_hyb_avg[i] = np.mean(n_hyb_indiv)
		n_hyb_sem[i] = ars.calcSEM(n_hyb_indiv)
	time = utils.getTime(nstep, dump_every)

	### plot
	ars.magicPlot()
	plt.figure(figLabel)
	plt.plot(time, n_hyb_avg)
	plt.fill_between(time, n_hyb_avg-n_hyb_sem, n_hyb_avg+n_hyb_sem, alpha=0.3)
	plt.axhline(y=n_scaf, color='k', linestyle='--')
	plt.xlabel("Time [s]")
	plt.ylabel("N\\textsubscript{hyb}")
	plt.ylim(-n_scaf*0.05, n_scaf*1.05)
	plt.title("Number of Scaffold Hybridizations")
	plt.legend(['Mean','SEM','N\\textsubscript{scaffold}'])


### plot number of hybridizations
def plotNmis(hyb_status_allSim, n_scaf, dump_every, misFile):

	### get misbinding energies
	Us_mis = utils.readMis(misFile)[1]
	nmisBond = len(Us_mis)

	### calculate free staple concentrations
	nsim = hyb_status_allSim.shape[0]
	nstep = hyb_status_allSim.shape[1]
	n_hyb_avg = np.zeros((nmisBond,nstep))
	n_hyb_sem = np.zeros((nmisBond,nstep))
	for m in range(nmisBond):
		for i in range(nstep):
			hyb_val = 1+(m+1)/100
			n_hyb_indiv = np.sum(hyb_status_allSim[:,i,:n_scaf]==hyb_val,axis=1)
			n_hyb_avg[m,i] = np.mean(n_hyb_indiv)
			n_hyb_sem[m,i] = ars.calcSEM(n_hyb_indiv)
	time = utils.getTime(nstep, dump_every)

	### plot
	ars.magicPlot()
	plt.figure("nMis")
	for m in range(nmisBond):
		plt.plot(time, n_hyb_avg[m], label=f'$U = {Us_mis[m]:0.2f}$')
		plt.fill_between(time, n_hyb_avg[m]-n_hyb_sem[m], n_hyb_avg[m]+n_hyb_sem[m], alpha=0.3)
	plt.xlabel("Time [s]")
	plt.ylabel("N\\textsubscript{mis}")
	plt.title("Number of Scaffold Misbonds")
	plt.legend()


################################################################################
### File Handlers

def readConn(connFile):
	ars.testFileExist(connFile, "connectivity")
	with open(connFile, 'rb') as f:
		params = pickle.load(f)
	strands = params['strands']
	n_scaf = params['n_scaf']
	nbead = params['nbead']
	return strands, n_scaf, nbead


### run the script
if __name__ == "__main__":
	main()
	print()

