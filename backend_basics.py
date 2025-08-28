import armament as ars
import utils
import argparse
import multiprocessing
from multiprocessing import Pool
import numpy as np
import pickle
import sys

## Description
# this script reads the geometry and trajectory of one or more simulations
  # and performs various useful analyses, namely trajectory centering and 
  # hybridization status calculation.
# all identity and connectivity information (strands, bonds, complements), comes
  # from the geometry file, whereas all dimensional information (positions, dbox)
  # comes from the trajectory file
# both atom indexing and strand indexing start at 1 (not 0), which may seem
  # like a poor choice given this python code, but it makes printing geometries
  # and atom dumps much easier.


################################################################################
### Parameters

def main():

	### get arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--copiesFile',		type=str,	default=None,	help='name of copies file (first column - simulation folder names)')	
	parser.add_argument('--simFold',		type=str,	default=None,	help='name of simulation folder, used if no copies file, defaults to current folder')
	parser.add_argument('--geoFileName',	type=str,	default=None,	help='name of geometry file (default to geometry.in)')
	parser.add_argument('--datFileName',	type=str,	default=None,	help='name of trajectory file (default to trajectory.dat)')
	parser.add_argument('--nstep_skip',		type=float,	default=0,		help='number of recorded initial steps to skip')
	parser.add_argument('--coarse_time',	type=int,	default=1,		help='coarse factor for time steps')
	parser.add_argument('--nstep_max',		type=float,	default=0,		help='max number of recorded steps to use, excluding initial frame (0 for all)')
	parser.add_argument('--misFile',		type=str,	default=None,	help='name of misbinding file, which contains cutoffs and energies')

	### analysis parameters
	center = 1						# what to place at center (1 for scaffold)
	unwrap = True					# whether to unwrap by strands at boundary
	bicolor = True					# whether to use only 2 colors (scaf and stap)
	r12_cut_hyb = 2.0				# hybridization potential cutoff radius

	### set arguments
	args = parser.parse_args()
	copiesFile = args.copiesFile
	simFold = args.simFold
	geoFileName = args.geoFileName
	datFileName = args.datFileName
	nstep_skip = int(args.nstep_skip)
	coarse_time = args.coarse_time
	nstep_max = int(args.nstep_max)
	misFile = args.misFile

	### interpret input
	if nstep_max == 0:
		nstep_max = 'all'
	if geoFileName is None:
		geoFileName = "geometry.in"
	if datFileName is None:
		datFileName = "trajectory.dat"


################################################################################
### Heart

	### get simulation folders
	simFolds, nsim = utils.getSimFolds(copiesFile, simFold)

	### read geometry
	geoFile = simFolds[0] + geoFileName
	strands, bonds_backbone, complements, compFactors, n_scaf, nbead, circularScaf = processGeo(geoFile)

	### write conectivity vars
	ars.createSafeFold("analysis")
	outConnFile = "analysis/connectivity_vars.pkl"
	writeConn(outConnFile, strands, bonds_backbone, complements, compFactors, n_scaf, nbead, circularScaf)

	### assemble data params
	params = []
	for i in range(nsim):
		params.append((simFolds[i], datFileName, misFile, strands, bonds_backbone, complements, compFactors, nstep_skip, nstep_max, coarse_time, center, unwrap, bicolor, r12_cut_hyb))

	### run in parallel if multiple simulations
	if nsim == 1:
		submain(*params[0])
	else:
		ncpu = min([nsim,multiprocessing.cpu_count()-1])
		with Pool(ncpu) as pool:
			pool.starmap(submain,params)


### body of main function
def submain(simFold, datFileName, misFile, strands, bonds_backbone, complements, compFactors, nstep_skip, nstep_max, coarse_time, center, unwrap, bicolor, r12_cut_hyb):

	### output files
	outFold = simFold + "analysis/"
	outDatFile = outFold + "trajectory_centered.dat"
	outGeoVisFile = outFold + "geometry_vis.in"
	outHybFile = outFold + "hyb_status.dat"

	### create analysis folder
	ars.createSafeFold(outFold)

	### figure out number of scaffold beads
	n_scaf = sum(strands==1)

	### read trajectory
	datFile = simFold + datFileName
	if ars.isinteger(nstep_max): nstep_max += 1
	points, _, dbox, used_every  = ars.readAtomDump(datFile, nstep_skip, coarse_time, nstep_max=nstep_max, getUsedEvery=True)

	### manipulate trajectory
	points_centered = ars.centerPointsMolecule(points, strands, dbox, center, unwrap)
	colors = np.minimum(2,strands) if bicolor else strands

	### write geometry and trajectory
	ars.writeGeo(outGeoVisFile, dbox, points_centered[0], strands, colors, bonds_backbone)
	writeAtomDump(outDatFile, dbox, points_centered, strands, used_every)

	### hybridization analysis
	hyb_status = calcHybStatus(points, dbox, n_scaf, misFile, complements, compFactors, r12_cut_hyb)
	writeHybStatus(outHybFile, hyb_status, used_every)


################################################################################
### File Handlers

### extract information from geometry file
def processGeo(geoFile):

	### read file
	_, molecules, types, _, bonds, _, extras = ars.readGeo(geoFile, extraLabel="Extras")

	### bead numbers and trimming
	n_scaf = sum(types==1)
	nbead = len(types)-1
	strands = molecules[:nbead]
	compFactors = extras[:nbead]

	### calculate complements
	complements = getComplements(extras, n_scaf, nbead)

	### get backbone bonds
	bonds_backbone = bonds[ (bonds[:,0]==1) | (bonds[:,0]==3) ]

	### determine if scaffold is circular
	circularScaf = True if np.sum(bonds_backbone[:,1]<=n_scaf)==n_scaf else False

	### return results
	return strands, bonds_backbone, complements, compFactors, n_scaf, nbead, circularScaf


### write connectivity vars pickle file
def writeConn(outConnFile, strands, bonds_backbone, complements, compFactors, n_scaf, nbead, circularScaf):
	params = {}
	params['strands'] = strands
	params['bonds_backbone'] = bonds_backbone
	params['complements'] = complements
	params['compFactors'] = compFactors
	params['n_scaf'] = n_scaf
	params['nbead'] = nbead
	params['circularScaf'] = circularScaf
	with open(outConnFile, 'wb') as f:
		pickle.dump(params, f)


### write lammps-style atom dump
def writeAtomDump(outDatFile, dbox, points, col2s, dump_every):
	nstep = points.shape[0]
	npoint = points.shape[1]
	len_npoint = len(str(npoint))
	len_ncol2 = len(str(max(col2s)))
	len_dbox = len(str(int(dbox/2)))
	with open(outDatFile, 'w') as f:
		for i in range(nstep):
			f.write(f"ITEM: TIMESTEP\n{i*dump_every}\n")
			f.write(f"ITEM: NUMBER OF ATOMS\n{npoint}\n")
			f.write(f"ITEM: BOX BOUNDS pp pp pp\n")
			f.write(f"-{dbox/2:0{len_dbox+3}.2f} {dbox/2:0{len_dbox+3}.2f} xlo xhi\n")
			f.write(f"-{dbox/2:0{len_dbox+3}.2f} {dbox/2:0{len_dbox+3}.2f} ylo yhi\n")
			f.write(f"-{dbox/2:0{len_dbox+3}.2f} {dbox/2:0{len_dbox+3}.2f} zlo zhi\n")
			f.write("ITEM: ATOMS id mol xs ys zs\n")
			for j in range(npoint):
				f.write(f"{j+1:<{len_npoint}} " + \
						f"{col2s[j]:<{len_ncol2}}  " + \
						f"{points[i,j,0]/dbox+1/2:11.8f} " + \
						f"{points[i,j,1]/dbox+1/2:11.8f} " + \
						f"{points[i,j,2]/dbox+1/2:11.8f}\n")


### write hybridization status file
def writeHybStatus(outHybFile, hyb_status, dump_every):
	nstep = hyb_status.shape[0]
	nbead = hyb_status.shape[1]
	with open(outHybFile, 'w') as f:
		for i in range(nstep):
			f.write(f"TIMESTEP {i*dump_every}\n")
			for j in range(nbead):
				f.write(f"{j+1} {hyb_status[i,j]}\n")


################################################################################
### Calculation Managers

### use charges to get complimentary beads (for all beads)
def getComplements(comp_tags, n_scaf, nbead):
	complements = [[] for i in range(nbead)]
	for i in range(n_scaf):
		for j in range(n_scaf,nbead):
			if all(comp_tags[i] == comp_tags[j]):
				complements[i].append(j+1)
	for i in range(n_scaf,nbead):
		for j in range(n_scaf):
			if all(comp_tags[i] == comp_tags[j]):
				complements[i].append(j+1)
	return complements


### analyze trajectory to get hybridiazation status of each bead at every time step
def calcHybStatus(points, dbox, n_scaf, misFile, complements, compFactors, r12_cut_hyb):
	nstep = points.shape[0]
	nbead = points.shape[1]
	hyb_status = np.zeros((nstep,nbead))

	### no misbinding
	if misFile is None:
		# 1 for hybridized
		# 0 for unhybridized
		# -1 for no complement

		for i in range(nstep):
			for j in range(nbead):
				if len(complements[j]) == 0:
					hyb_status[i,j] = -1
				elif not all(points[i,j,:]==[0,0,0]):
					for k in range(len(complements[j])):
						c = complements[j][k]-1
						if not all(points[i,c,:]==[0,0,0]):
							sep = np.linalg.norm( ars.applyPBC( points[i,j,:]-points[i,c,:], dbox ) )
							if sep < r12_cut_hyb:
								hyb_status[i,j] = 1

	### misbinding
	else:
		# 1 for hybridized, decimal for level (1.00 for misbind, 1.01 for strongest misbond, etc...)
		# 0 for unhybridized

		mis_d2_cuts = utils.readMis(misFile)[0]
		nmisBond = len(mis_d2_cuts)
		for i in range(nstep):
			for j in range(n_scaf):
				for k in range(n_scaf,nbead):
					if not all(points[i,k,:]==[0,0,0]):
						sep = np.linalg.norm( ars.applyPBC( points[i,j,:]-points[i,k,:], dbox ) )
						if sep < r12_cut_hyb:
							d2 = sum((compFactors[j]-compFactors[k])**2)
							if d2 == 0:
								hyb_status[i,j] = 1
								hyb_status[i,k] = 1
							elif d2 < mis_d2_cuts[-1]:
								level = np.searchsorted(mis_d2_cuts, d2)
								hyb_status[i,j] = 1 + (level+1)/100
								hyb_status[i,k] = 1 + (level+1)/100

	### result
	return hyb_status


### run the script
if __name__ == "__main__":
	main()
	print()

