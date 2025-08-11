import armament as ars
import utils
import numpy as np
import argparse
import random
import os
import sys

## Description
# this script reads the output from a previous dnafold_lmp simulation and writes
  # the files necessary for restarting the simulation.
# if adding staples, the restart geometry is used; otherwise, the binary restart
  # file is used.
# unlike most other scripts in this package, this script cannot read a copies file
  # and perform its operation on a batch of copied simulations; this is due to the
  # difficulty and unintuitiveness of matching simulations with random seeds, which
  # would neccesitate a new copies file that included associated random seeds.


################################################################################
### Parameters

def main():

	### get arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--nstep',		type=float,	required=True,	help='number of simulation steps')
	parser.add_argument('--simFold',	type=str,	default=None,	help='name of simulation folder, should exist within current directory')
	parser.add_argument('--rseed',		type=int,	default=1,		help='random seed, used to find simFold if necessary')
	parser.add_argument('--astapFile',	type=str,	default=None,	help='if adding staples, path to add staple file')

	### parameters
	r12_eq = 2.72				# if adding staples, equilibrium bead separation
	sigma = 2.14				# if adding staples, bead Van der Waals radius

	### set arguments
	args = parser.parse_args()
	nstep = int(args.nstep)
	simFold = args.simFold
	rseed = args.rseed
	astapFile = args.astapFile

	### get simulation folders
	simFolds, nsim = utils.getSimFolds(None, simFold, rseed)


################################################################################
### Heart

	### loop over simulations
	for i in range(nsim):

		### edit input file
		lammpsFile = simFolds[i] + "lammps.in"
		nbondType = editInput(lammpsFile, nstep, astapFile)

		### add staples
		if astapFile is not None:

			### read geometry file
			geoFile = simFolds[i] + "restart_geometry.out"
			r, molecules, types, charges, bonds, angles, extras = ars.readGeo(geoFile, extraLabel="CFs")
			dbox = ars.getDbox3(geoFile)[0]
			strands = molecules[:-1]

			### create random number generator
			rng = np.random.default_rng(rseed)

			### add staples
			nstrand = max(strands)
			is_add_strand = readAstap(astapFile, nstrand)
			r, types = addStap(r, strands, types, is_add_strand, r12_eq, sigma, dbox, rng)
			bonds = updateBondTypes(bonds, types)

			### write geometry file
			outGeoFile = simFolds[i] + "restart_geometry.in"
			ars.writeGeo(outGeoFile, dbox, r, molecules, types, bonds, angles, nbondType=nbondType, nangleType=2, charges=charges, extras=extras)


################################################################################
### File Handlers

### read old lammps file and write 
def editInput(lammpsFile, nstep, astapFile):

	### read old lammps file
	ars.testFileExist(lammpsFile)
	with open(lammpsFile,'r') as f:
		content_in = f.readlines()

	### parse the content and write edited content
	content_out = []
	nbondType = 0
	i = 0
	while i < len(content_in):
		content_out.append(content_in[i])

		### set how to read geometry
		if content_in[i].startswith("## Geometry"):

			### get fix CFs line
			if "read_restart" in content_in[i+1]:
				line_fixCFs = content_in[i+2]
			else:
				line_fixCFs = content_in[i+1]

			### if adding stales, use geometry
			if astapFile is not None:
				content_out.append(line_fixCFs)
				content_out.append("read_data       restart_geometry.in fix CFs NULL Extras &\n")
				content_out.append("                extra/bond/per/atom 10 &\n")
				content_out.append("                extra/angle/per/atom 10 &\n")
				content_out.append("                extra/special/per/atom 100\n")

			### otherwise, use binary
			else:
				content_out.append("read_restart    restart_binary2.out\n")
				content_out.append(line_fixCFs)
				
			### skip the appropriate number of lines
			if "read_restart" in content_in[i+1]:
				i += 2
			else:
				i += 5

		### remove scaffold relaxation
		elif content_in[i].startswith("## Relaxation"):
			content_out.pop()
			i += 10

		### run a single step before dumps
		elif content_in[i].startswith("## Production"):
			if "run" not in content_in[i+4]:
				content_out.append(content_in[i+1])
				content_out.append(content_in[i+2])
				content_out.append(content_in[i+3])
				content_out.append("run             1\n")
				i += 3

		### adjust run time
		elif content_in[i].startswith("## Go Time"):
			content_out.append(f"run             {int(nstep-1)}\n")
			i += 1

		### count bond types
		elif content_in[i].startswith("bond_coeff"):
			bondType = int(content_in[i].split()[1])
			nbondType = max(bondType,nbondType)

		### next line
		i += 1

	### write edited lammps file
	with open(lammpsFile,'w') as f:
		f.writelines(content_out)

	### result
	return nbondType


### read add staple file
def readAstap(astapFile, nstrand):
	is_add_strand = [ False for i in range(nstrand) ]

	### read add staples file
	ars.testFileExist(astapFile,"add staples")
	with open(astapFile,'r') as f:
		add_strands = [ int(line.strip()) for line in f ]
	for si in range(len(add_strands)):
		is_add_strand[add_strands[si]-1] = True

	### return strand addition status
	return is_add_strand


################################################################################
### Calculation Managers

### edit positions and types for adding staples
def addStap(r, strands, types, is_add_strand, r12_eq, sigma, dbox, rng):
	print("Initializing positions...")

	### parameters
	max_nfail_strand = 20
	max_nfail_bead = 20
	nbead = sum(strands>0)
	n_scaf = sum(strands==1)
	is_active = types!=3

	### initializations
	nbead_placed = n_scaf
	nbead_locked = n_scaf
	nstrand_locked = 1
	nstrand_placed = 0

	### loop over beads
	nfail_strand = 0
	while nbead_placed < nbead:
		bi = nbead_placed

		### skip strands that don't need adding
		if not is_add_strand[strands[bi]-1] or is_active[bi]:

			### warning if trying to add active strand
			if is_add_strand[strands[bi]-1] and is_active[bi]:
				print("Flag: skipping the addition of an already active staple.")

			### updates
			nbead_placed += sum(strands==nstrand_locked+1)
			nbead_locked += sum(strands==nstrand_locked+1)
			nstrand_locked += 1
			continue

		### attempt to place bead
		nfail_bead = 0
		while True:

			### position linked to previous bead
			if strands[bi] == strands[bi-1]:
				r_propose = ars.applyPBC(r[bi-1] + r12_eq*ars.unitVector(ars.boxMuller(rng)), dbox)

			### random position for new strand
			else:
				r_propose = ars.randPos(dbox, rng)

			### evaluate position, break loop if no overlap
			if not ars.checkOverlap(r_propose,r[:bi],sigma,dbox):
				break

			### if loop not broken update bead fail count
			nfail_bead += 1

			### break bead loop if too much failure
			if nfail_bead == max_nfail_bead:
				break

		### set position if all went well
		if nfail_bead < max_nfail_bead:
			r[bi] = r_propose
			nbead_placed += 1
			types[bi] = 2

			### update locked strands if end of strand
			if bi+1 == nbead or strands[bi] < strands[bi+1]:
				nbead_locked += sum(strands==nstrand_locked+1)
				nstrand_locked += 1
				nstrand_placed += 1
				nfail_strand = 0

		### reset strand if too much failure
		else:
			nfail_strand += 1
			nbead_placed = nbead_locked

			### give up if too many strand failures
			if nfail_strand == max_nfail_strand:
				print("Error: could not place beads, try again with larger box.")
				sys.exit()

	### return positions
	print(f"Placed {nstrand_placed} staples...")
	return r, types


### activate bonds based on bead types
def updateBondTypes(bonds, types):
	nbond = len(bonds)
	for i in range(nbond):
		if types[bonds[i,1]-1] == 2 and types[bonds[i,2]-1] == 2:
			bonds[i,0] = 1
	return bonds


### run the script
if __name__ == "__main__":
	main()
	print()

