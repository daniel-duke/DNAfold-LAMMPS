import armament as ars
import utils
import argparse
from ovito import scene
from ovito.io import import_file
from ovito.vis import Viewport, SimulationCellVis
from ovito.modifiers import LoadTrajectoryModifier
from ovito.modifiers import ComputePropertyModifier
from ovito.modifiers import ColorCodingModifier
from ovito.modifiers import DeleteSelectedModifier
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

## Description
# this script reads a DNAfold hybridization time trajectory (or a batch of
  # trajectories), calculates the first hybridization times, and writes 
  # geometry, trajectory, and ovito files for visualizing these results.
# this script will only work if "backend_basics.py" has already been run for
  # the given simulation (requires a populated "analysis" folder).
# there are 4 hybridization time results:
  # 1) individual geometry: first hyb time 
  # 2) individual trajectory: hyb status
  # 3) averaged geometry: averaged first hyb time
  # 4) averaged rajectory: hyb status probability


################################################################################
### Parameters

def main():

	### get arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--copiesFile',	type=str,	default=None,	help='name of copies file, which contains a list of simulation folders')	
	parser.add_argument('--simFold',	type=str,	default=None,	help='name of simulation folder, should exist within current directory')
	parser.add_argument('--rseed',		type=int,	default=1,		help='random seed, used to find simFold if necessary')
	parser.add_argument('--cadFile',	type=str,	required=True,	help='name of caDNAno file, for initializimg positions')
	parser.add_argument('--topFile',	type=str, 	default=None,	help='if using oxdna positions, name of topology file')
	parser.add_argument('--confFile',	type=str, 	default=None,	help='if using oxdna positions, name of conformation file')
	parser.add_argument('--writeIndiv',	type=int,	default=True,	help='whether to write individual hybridization trajectories')
	parser.add_argument('--includeMis',	type=int,	default=False,	help='whether to count misbonds as hybridizations')
	parser.add_argument('--avg_type',	type=str,	default="last",	help='how to get hyb times for each staple (values: direct, avg, first, last)')

	### set arguments
	args = parser.parse_args()
	cadFile = args.cadFile
	copiesFile = args.copiesFile
	simFold = args.simFold
	rseed = args.rseed
	topFile = args.topFile
	confFile = args.confFile
	writeIndiv = args.writeIndiv
	includeMis = args.includeMis
	avg_type = args.avg_type

	### check input
	if topFile is not None and confFile is not None:
		position_src = "oxdna"
	else:
		position_src = "cadnano"
		if topFile is not None:
			print("Flag: oxDNA topology file provided without configuration file, using caDNAno positions.")
		if confFile is not None:
			print("Flag: oxDNA configuration file provided without topology file, using caDNAno positions.")


################################################################################
### Heart

	### get simulation folders
	simFolds, nsim = utils.getSimFolds(copiesFile, simFold, rseed)

	### get pickled data
	connFile = "analysis/connectivity_vars.pkl"
	strands, bonds_backbone, complements, n_scaf, nbead = readConn(connFile)

	### prepare position data
	if position_src == "cadnano":
		r = utils.initPositionsCaDNAno(cadFile)[0]
	if position_src == "oxdna":
		r = utils.initPositionsOxDNA(cadFile, topFile, confFile)[0]
	dbox3 = prepGeoData(r)

	### get minimum number of steps
	nstep_allSim = np.zeros(nsim,dtype=int)
	for i in range(nsim):
		datFile = simFolds[i] + "analysis/trajectory_centered.dat"
		nstep_allSim[i] = ars.getNstep(datFile)
	nstep_min = int(min(nstep_allSim))

	### get hybridization dump frequency
	hybFile = simFolds[0] + "analysis/hyb_status.dat"
	dump_every = utils.getDumpEveryHyb(hybFile)

	### determine how to handle misbonds
	mis_status = 'hyb' if if includeMis else 'none'

	### loop over simulations
	hyb_status_allSim = np.zeros((nsim,nstep_min,nbead))
	first_hyb_times_allSim = np.zeros((nsim,n_scaf))
	for i in range(nsim):

		### analyze hybridizations
		hybFile = simFolds[i] + "analysis/hyb_status.dat"; print()
		hyb_status_allSim[i] = utils.readHybStatus(hybFile, nstep_max=nstep_min, mis_status=mis_status)
		first_hyb_times_allSim[i], first_hyb_times_scaled = utils.calcFirstHybTimes(hyb_status_allSim[i], complements, n_scaf, dump_every)

		### write hybridization trajectory
		if writeIndiv:
			outGeoFile = simFolds[i] + "analysis/hyb_times_geometry.in"
			outDatFile = simFolds[i] + "analysis/hyb_times_trajectory.dat"
			ovitoFile = simFolds[i] + "analysis/vis_hyb_times.ovito"
			colors = propScafToStap(first_hyb_times_scaled, complements, strands, avg_type)
			ars.writeGeo(outGeoFile, dbox3, r, types=strands, charges=colors, bonds=bonds_backbone)
			colors = propScafToStap(hyb_status_allSim[i][:,:n_scaf], complements, strands, avg_type)
			writeAtomDump(outDatFile, dbox3, r, colors, dump_every)
			writeOvito(ovitoFile, outGeoFile, outDatFile)

	### averaged binding times
	if nsim > 1:
		outGeoFile = "analysis/hyb_times_geometry.in"
		outDatFile = "analysis/hyb_times_trajectory.dat"
		ovitoFile = "analysis/vis_hyb_times.ovito"
		hyb_status_avg, first_hyb_times_scaled_avg = averageHybData(hyb_status_allSim, first_hyb_times_allSim, dump_every)
		colors = propScafToStap(first_hyb_times_scaled_avg, complements, strands, avg_type)
		ars.writeGeo(outGeoFile, dbox3, r, types=strands, charges=colors, bonds=bonds_backbone)
		colors = propScafToStap(hyb_status_avg[:,:n_scaf], complements, strands, avg_type)
		writeAtomDump(outDatFile, dbox3, r, colors, dump_every)
		writeOvito(ovitoFile, outGeoFile, outDatFile)


################################################################################
### File Handlers

### write lammps-style trajectory for stationairy points with changing colors
def writeAtomDump(outDatFile, dbox3, r, colors, dump_every):
	nstep = colors.shape[0]
	npoint = r.shape[0]
	len_npoint = len(str(npoint))
	len_ncolor = len(str(max([max(i) for i in colors])))
	len_dbox3 = len(str(int(max(dbox3)/2)))
	with open(outDatFile,'w') as f:
		for i in range(nstep):
			f.write(f"ITEM: TIMESTEP\n{i}\n")
			f.write(f"ITEM: NUMBER OF ATOMS\n{npoint}\n")
			f.write(f"ITEM: BOX BOUNDS pp pp pp\n")
			f.write(f"-{dbox3[0]/2:0{len_dbox3+3}.2f} {dbox3[0]/2:0{len_dbox3+3}.2f} xlo xhi\n")
			f.write(f"-{dbox3[1]/2:0{len_dbox3+3}.2f} {dbox3[1]/2:0{len_dbox3+3}.2f} ylo yhi\n")
			f.write(f"-{dbox3[2]/2:0{len_dbox3+3}.2f} {dbox3[2]/2:0{len_dbox3+3}.2f} zlo zhi\n")
			f.write(f"ITEM: ATOMS id q xs ys zs\n")
			for j in range(npoint):
				f.write(f"{j+1:<{len_npoint}} " + \
						f"{colors[i][j]:<{len_ncolor}}  " + \
						f"{r[j,0]/dbox3[0]+1/2:10.8f} " + \
						f"{r[j,1]/dbox3[1]+1/2:10.8f} " + \
						f"{r[j,2]/dbox3[2]+1/2:10.8f}\n")


### write session state ovito file that visualizes the geometry
def writeOvito(ovitoFile, outGeoFile, outDatFile):

	### set colors
	scaf_default_color = ars.getColor("purple")
	scaf_noComp_color = ars.getColor("grey")

	### initialize pipeline
	pipeline = import_file(outGeoFile, atom_style="full")
	pipeline.add_to_scene()

	### prepare basic DNAfold scene
	pipeline = utils.setOvitoBasics(pipeline)

	### load trajectory
	traj_pipeline = import_file(outDatFile, multiple_frames=True)
	traj_mod = LoadTrajectoryModifier()
	traj_mod.source.load(outDatFile)
	pipeline.modifiers.append(traj_mod)

	### set scaffold and staple particle radii and bond widths (small scaffold)
	pipeline.modifiers.append(ComputePropertyModifier(output_property='Radius', expressions=['(ParticleType==1)?0.6:1']))
	pipeline.modifiers.append(ComputePropertyModifier(operate_on='bonds', output_property='Width', expressions=['(@1.ParticleType==1)?1.2:2']))

	### set color coding
	pipeline.modifiers.append(ColorCodingModifier(property='Charge', start_value=0, end_value=1, gradient=ColorCodingModifier.Viridis()))
	pipeline.modifiers.append(ComputePropertyModifier(output_property='Color', expressions=[f'(ParticleType==1)?{scaf_default_color[0]}/255:Color.R', f'(ParticleType==1)?{scaf_default_color[1]}/255:Color.G', f'(ParticleType==1)?{scaf_default_color[2]}/255:Color.B']))
	pipeline.modifiers.append(ComputePropertyModifier(enabled=False, output_property='Color', expressions=[f'(Charge==-1)?{scaf_noComp_color[0]}/255:Color.R', f'(Charge==-1)?{scaf_noComp_color[1]}/255:Color.G', f'(Charge==-1)?{scaf_noComp_color[2]}/255:Color.B']))

	### add option to delete staples
	pipeline.modifiers.append(ComputePropertyModifier(enabled=False, output_property='Selection', expressions=['ParticleType!=1']))
	pipeline.modifiers.append(DeleteSelectedModifier())

	### write ovito file
	scene.save(ovitoFile)
	pipeline.remove_from_scene()


################################################################################
### File Handlers

def readConn(connFile):
	ars.testFileExist(connFile, "connectivity")
	with open(connFile, 'rb') as f:
		params = pickle.load(f)
	strands = params['strands']
	bonds_backbone = params['bonds_backbone']
	complements = params['complements']
	n_scaf = params['n_scaf']
	nbead = params['nbead']
	return strands, bonds_backbone, complements, n_scaf, nbead


################################################################################
### Calculation Managers

### average the first hyb times across several simulations
def averageHybData(hyb_status_allSim, first_hyb_times_allSim, dump_every):
	nsim = hyb_status_allSim.shape[0]
	nstep = hyb_status_allSim.shape[1]
	nbead = hyb_status_allSim.shape[2]
	n_scaf = first_hyb_times_allSim.shape[1]

	### deal with hyb status
	hyb_status_avg = np.mean(hyb_status_allSim, axis=0)

	### deal with first hyb times
	first_hyb_times_scaled_avg = np.zeros(n_scaf)
	for i in range(n_scaf):

		### if no complement, keep -1
		if first_hyb_times_allSim[0,i] == -1:
			first_hyb_times_scaled_avg[i] = -1
		elif all(first_hyb_times_allSim[:,i]==0):
			first_hyb_times_scaled_avg[i] = 1
		else:
			first_hyb_times_scaled_bead = first_hyb_times_allSim[:,i]/nstep/dump_every
			first_hyb_times_scaled_avg[i] = np.mean(first_hyb_times_scaled_bead[first_hyb_times_scaled_bead != 0])

	### results
	return hyb_status_avg, first_hyb_times_scaled_avg


### set property of scaffold beads to the complementary staple beads
def propScafToStap(prop_scaf, complements, strands, avg_type):

	### single time step
	if prop_scaf.ndim == 1:
		prop = propScafToStapSingle(prop_scaf, strands, complements, avg_type)

	### trajectory
	if prop_scaf.ndim == 2:
		nstep = prop_scaf.shape[0]
		n_ori = len(complements)
		prop = np.zeros((nstep,n_ori))
		for i in range(nstep):
			prop[i] = propScafToStapSingle(prop_scaf[i], strands, complements, avg_type)

	### result
	return prop


### calculate correlation between hybridization times (h) and product quality (q)
def propScafToStapSingle(prop_scaf, strands, complements, avg_type):
	n_scaf = len(prop_scaf)
	n_ori = len(complements)
	nstrand = max(strands)
	prop = np.zeros(n_ori)
	prop[:n_scaf] = prop_scaf

	### direct
	if avg_type == "direct":
		prop[:n_scaf] = prop_scaf
		for bi in range(n_scaf,n_ori):
			if len(complements[bi]) > 0:
				prop[bi] = prop_scaf[complements[bi][0]-1]

	### group correlation by strands, pre-correlation methods
	elif avg_type == "avg" or avg_type == "first" or avg_type == "last":
		prop_grouped = [ np.array([]) for si in range(nstrand) ]
		prop_grouped[0] = np.zeros(1)
		for bi in range(n_scaf):
			if len(complements[bi]) > 0:
				strand = strands[complements[bi][0]-1]
				prop_grouped[strand-1] = np.append(prop_grouped[strand-1],prop_scaf[bi])
		prop_strand = np.zeros(nstrand)
		for si in range(1,nstrand):
			if avg_type == "avg":
				prop_strand[si] = np.mean(prop_grouped[si])
			elif avg_type == "first":
				prop_strand[si] = np.min(prop_grouped[si])
			elif avg_type == "last":
				prop_strand[si] = np.max(prop_grouped[si])
		for bi in range(n_scaf,n_ori):
			prop[bi] = prop_strand[strands[bi]-1]

	### error message
	else:
		print("Error: Unknown average type.\n")
		sys.exit()

	### result
	return prop


################################################################################
### Utility Functions

### get geometry data ready for visualization
def prepGeoData(r):

	### box diameter
	dbox3 = [ max(abs(r[:,0]))+2.72, max(abs(r[:,1]))+2.4, max(abs(r[:,2]))+2.4 ]
	dbox3 = [ 2*i for i in dbox3 ]

	return dbox3


### run the script
if __name__ == "__main__":
	main()
	print()

