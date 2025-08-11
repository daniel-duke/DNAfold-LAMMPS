import armament as ars
import utils
import argparse
import multiprocessing
from multiprocessing import Pool
import numpy as np
import pickle
import sys

## Description
# this script reads the geometry and trajectory of a dnafold_lmp simulations
  # and performs various useful analyses, namely trajectory centering and first
  # bind times.
# all indexing starts at 1 (atom indices, strand indices, etc); this may seem
  # like a poor choice given this code is in python, but it makes printing
  # geometries and atom dumps much easier.

## File Descriptions
# geometry:
  # 3 atom types - (1) scaffold, (2) staple, (3) dummy
  # 3 bond types - (1) backbone, (2) hybridization, (3) dummy
  # 4 angle types - (1) off 180, (2) off 90, (3) on 180, (4) on 90
  # columns - (1) atom index, (2) strand index, (3) scaf/stap/dummy type, (4) charge, (5-7) position
# geometry_vis:
  # nstrand atom types - (1-nstrand) strand index
  # 1 bond type - (1) backbone
  # columns - (1) atom index, (2) zero, (3) strand index, (4-6) position
# trajectory:
  # columns - (1) atom index, (2) strand index, (3-5) scaled position


################################################################################
### Parameters

def main():

	### get arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--copiesFile',		type=str,	default=None,	help='name of copies file, which contains a list of simulation folders')	
	parser.add_argument('--simFold',		type=str,	default=None,	help='name of simulation folder, should exist within current directory')
	parser.add_argument('--rseed',			type=int,	default=1,		help='random seed, used to find simFold if necessary')
	parser.add_argument('--nstep_skip',		type=float,	default=0,		help='number of recorded initial steps to skip')
	parser.add_argument('--nstep_max',		type=float,	default=0,		help='max number of recorded steps to use (0 for all)')
	parser.add_argument('--coarse_time',	type=int,	default=1,		help='coarse factor for time steps')

	### analysis options
	center = 1						# what to place at center
	unwrap = 1						# whether to unwrap by strands at boundary
	set_color = 0					# whether to set the colors
	bicolor = True					# whether to use only 2 colors (scaf and stap)
	r12_cut_hyb = 2					# hybridization potential cutoff radius

	### set arguments
	args = parser.parse_args()
	copiesFile = args.copiesFile
	simFold = args.simFold
	rseed = args.rseed
	nstep_skip = int(args.nstep_skip)
	nstep_max = int(args.nstep_max)
	coarse_time = args.coarse_time

	### interpret input
	if nstep_max == 0:
		nstep_max = "all"


################################################################################
### Heart

	### get simulation folders
	simFolds, nsim = utils.getSimFolds(copiesFile, simFold, rseed)

	### get connectivity vars
	geoFile = simFolds[0] + "geometry.in"
	strands, bonds_backbone, complements, compFactors, n_scaf, nbead, circularScaf = processGeo(geoFile)

	### write conectivity vars
	ars.createSafeFold("analysis")
	outConnFile = "analysis/connectivity_vars.pkl"
	with open(outConnFile, 'wb') as f:
		pickle.dump([strands, bonds_backbone, complements, n_scaf, nbead, circularScaf], f)

	### assemble data params
	params = []
	for i in range(nsim):
		params.append((simFolds[i], strands, bonds_backbone, complements, compFactors, nstep_skip, nstep_max, coarse_time, center, unwrap, set_color, bicolor, r12_cut_hyb))

	### run in parallel if multiple simulations
	if nsim == 1:
		submain(*params[0])
	else:
		ncpu = min([nsim,multiprocessing.cpu_count()-1])
		with Pool(ncpu) as pool:
			pool.starmap(submain,params)


### body of main function
def submain(simFold, strands, bonds_backbone, complements, compFactors, nstep_skip, nstep_max, coarse_time, center, unwrap, set_color, bicolor, r12_cut_hyb):

	### input files
	datFile = simFold + "trajectory.dat"
	misFile = "misbinding.txt"

	### output files
	outFold = simFold + "analysis/"
	outDatFile = outFold + "trajectory_centered.dat"
	outGeoVisFile = outFold + "geometry_vis.in"
	outHybFile = outFold + "hyb_status.dat"

	### create analysis folder
	ars.createSafeFold(outFold)

	### trajectory centering
	points, strands, dbox = ars.readAtomDump(datFile, nstep_skip, coarse_time, nstep_max=nstep_max);
	dump_every = ars.getDumpEvery(datFile)*coarse_time
	points_centered = ars.centerPointsMolecule(points, strands, dbox, center, unwrap)
	colors = np.minimum(2,strands) if bicolor else strands

	### write complete trajectories
	writeAtomDump(outDatFile, dbox, points_centered, colors, set_color, dump_every)
	ars.writeGeo(outGeoVisFile, dbox, points[0], strands, colors, bonds_backbone)

	### read misbinding data
	misbinding = False
	if compFactors.shape[1] > 1:
		if ars.testFileExist(misFile, "misbinding", required=False):
			misbinding = True
			mis_d2_cuts, Us_mis = readMisbinding(misFile)
			n_scaf = sum(strands==1)
		else:
			print("Flag: Simulation seems to include misbinding, but no misbinding data file found.")

	### hybridization analysis
	if misbinding: hyb_status = calcHybStatusMis(points, compFactors, n_scaf, dbox, r12_cut_hyb, mis_d2_cuts, Us_mis)
	if not misbinding: hyb_status = calcHybStatus(points, complements, dbox, r12_cut_hyb)
	writeHybStatus(outHybFile, hyb_status, dump_every)


################################################################################
### File Managers

### extract information from geometry file
def processGeo(geoFile):

	### read file
	_, molecules, types, _, bonds, _, extras = ars.readGeo(geoFile, extraLabel="Extras")

	### bead numbers and trimming
	n_scaf = sum(types==1)
	nbead = len(types)-1
	strands = molecules[:-1]
	compFactors = extras[:-1]

	### calculate complements
	complements = getComplements(extras, n_scaf, nbead)

	### analyze bonds
	bonds_backbone = bonds[ (bonds[:,0]!=2) & (bonds[:,1]<=nbead) & (bonds[:,2]<=nbead) ]
	circularScaf = True if np.sum(bonds_backbone[:,1]<=n_scaf)==n_scaf else False

	### return results
	return strands, bonds_backbone, complements, compFactors, n_scaf, nbead, circularScaf


### extract misbinding info
def readMisbinding(misFile):
	with open(misFile, 'r') as f:
		content = f.readlines()
	nmisBond = len(content)
	mis_d2_cuts = np.zeros(nmisBond)
	Us_mis = np.zeros(nmisBond)
	for i in range(nmisBond):
		line = content[i].split()
		mis_d2_cuts[i] = line[0]
		Us_mis[i] = line[1]
	return mis_d2_cuts, Us_mis


### write lammps-style atom dump
def writeAtomDump(outDatFile, dbox, points, col2s, set_color, dump_every):
	nstep = points.shape[0]
	npoint = points.shape[1]
	len_npoint = len(str(npoint))
	len_ncol2 = len(str(max(col2s)))
	len_dbox = len(str(int(dbox)))
	with open(outDatFile, 'w') as f:
		for i in range(nstep):
			f.write(f"ITEM: TIMESTEP\n{i*dump_every}\n")
			f.write(f"ITEM: NUMBER OF ATOMS\n{npoint}\n")
			f.write(f"ITEM: BOX BOUNDS pp pp pp\n")
			f.write(f"-{dbox/2:0{len_dbox+3}.2f} {dbox/2:0{len_dbox+3}.2f} xlo xhi\n")
			f.write(f"-{dbox/2:0{len_dbox+3}.2f} {dbox/2:0{len_dbox+3}.2f} ylo yhi\n")
			f.write(f"-{dbox/2:0{len_dbox+3}.2f} {dbox/2:0{len_dbox+3}.2f} zlo zhi\n")
			if set_color:
				f.write("ITEM: ATOMS id type xs ys zs\n")
			else:
				f.write("ITEM: ATOMS id mol xs ys zs\n")
			for j in range(npoint):
				f.write(f"{j+1:<{len_npoint}} " + \
						f"{col2s[j]:<{len_ncol2}}  " + \
						f"{points[i,j,0]/dbox+1/2:10.8f} " + \
						f"{points[i,j,1]/dbox+1/2:10.8f} " + \
						f"{points[i,j,2]/dbox+1/2:10.8f}\n")


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


### analyze trajectory to get hybridiazation status of each scaffold bead for each time step
# 1 for hybridized
# 0 for unhybridized
# -1 for no complement
def calcHybStatus(points, complements, dbox, r12_cut_hyb):
	r12_cut_hyb = 2
	nstep = points.shape[0]
	nbead = points.shape[1]
	hyb_status = np.zeros((nstep,nbead),dtype=int)
	for i in range(nstep):
		for j in range(nbead):
			if len(complements[j]) == 0:
				hyb_status[i,j] = -1
			elif any(points[i,j,:]!=[0,0,0]):
				for k in range(len(complements[j])):
					c = complements[j][k]-1
					if any(points[i,c,:]!=[0,0,0]):
						sep = np.linalg.norm( ars.applyPBC( points[i,j,:]-points[i,c,:], dbox ) )
						if sep < r12_cut_hyb:
							hyb_status[i,j] = 1
	### result
	return hyb_status


### analyze trajectory to get hybridiazation status of each scaffold bead for each time step
# 1 for hybridized, decimal for U
# 0 for unhybridized
def calcHybStatusMis(points, compFactors, n_scaf, dbox, r12_cut_hyb, mis_d2_cuts, Us_mis):
	r12_cut_hyb = 2
	nstep = points.shape[0]
	nbead = points.shape[1]
	nmisBond = len(mis_d2_cuts)
	hyb_status = np.zeros((nstep,nbead))
	for i in range(nstep):
		for j in range(n_scaf):
			if any(points[i,j,:]!=[0,0,0]):
				for k in range(n_scaf,nbead):
					if any(points[i,k,:]!=[0,0,0]):
						sep = np.linalg.norm( ars.applyPBC( points[i,j,:]-points[i,k,:], dbox ) )
						if sep < r12_cut_hyb:
							d2 = sum((compFactors[j]-compFactors[k])**2)
							if d2 == 0:
								hyb_status[i,j] = 1
								hyb_status[i,k] = 1
							elif d2 < mis_d2_cuts[-1]:
								for m in range(nmisBond):
									if d2 < mis_d2_cuts[m]:
										hyb_status[i,j] = 1 + (m+1)/100
										hyb_status[i,k] = 1 + (m+1)/100
										break

	### result
	return hyb_status


### run the script
if __name__ == "__main__":
	main()
	print()
