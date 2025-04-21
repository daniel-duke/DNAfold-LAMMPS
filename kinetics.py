import armament as ars
import utils
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

## Description
# this script reads a hybridization status file for a DNAfold simulation
  # and calculates the free staple kinetics.
# this script will only work if "backend_basics.py" has already been run for
  # the given simulation (requires a populated "analysis" folder).


################################################################################
### Parameters

def main():

	### get arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--copiesFile', type=str, required=True, help='name of copies file, which contains a list of simulation folders')	

	### set arguments
	args = parser.parse_args()
	copiesFile = args.copiesFile

	### get simulation folders
	simFolds, nsim = utils.getSimFolds(copiesFile)


################################################################################
### Heart

	### get pickled data
	connFile = "analysis/connectivity_vars.pkl"
	ars.testFileExist(connFile, "connectivity")
	with open(connFile, 'rb') as f:
		[strands, _, _, n_scaf, nbead] = pickle.load(f)[:5]

	### get minimum number of steps
	nstep_allSim = np.zeros(nsim,dtype=int)
	for i in range(nsim):
		datFile = simFolds[i] + "analysis/trajectory_centered.dat"
		nstep_allSim[i] = ars.getNstep(datFile)
	nstep_min = int(min(nstep_allSim))

	### loop over simulations
	hyb_status_allSim = np.zeros((nsim,nstep_min,nbead))
	for i in range(nsim):

		### analyze hybridizations
		hybFile = simFolds[i] + "analysis/hyb_status.dat"
		hyb_status_allSim[i], dump_every = utils.readHybStatus(hybFile, nstep_min)

	### analyze kinetics
	plotKinetics(hyb_status_allSim, strands, n_scaf, dump_every)
	plt.show()


################################################################################
### Plotters

### plot free staple kinetics
def plotKinetics(hyb_status_allSim, strands, n_scaf, dump_every):

	### calculate free staple concentrations
	nsim = hyb_status_allSim.shape[0]
	nstep = hyb_status_allSim.shape[1]
	nbead = hyb_status_allSim.shape[2]
	nstap = max(strands)-1
	conc_avg = np.zeros(nstep)
	conc_sem = np.zeros(nstep)
	for i in range(nstep):
		stap_freedom = np.ones((nstap,nsim),dtype=int)
		for j in range(n_scaf,nbead):
			for k in range(nsim):
				if hyb_status_allSim[k,i,j] == 1:
					stap_freedom[strands[j]-2,k] = 0
		conc_indiv = np.mean(stap_freedom,axis=1)
		conc_avg[i] = max(np.mean(conc_indiv), 1/nstap/nsim)
		conc_sem[i] = ars.calcSEM(conc_indiv)

	### calculate time
	dt = 0.01
	scale = 5200
	time = np.arange(nstep)*dump_every*dt*scale*1E-9

	### plot
	ars.magicPlot()
	plt.figure("Kinetics",figsize=(8,6))
	plt.plot(time,np.log(conc_avg))
	plt.fill_between(time,np.log(conc_avg-conc_sem),np.log(conc_avg+conc_sem),alpha=0.3)
	plt.xlabel("Time [s]")
	plt.ylabel("$\\ln(C/C_0)$")
	plt.title("Free Staple Kinetics")


### run the script
if __name__ == "__main__":
	main()

