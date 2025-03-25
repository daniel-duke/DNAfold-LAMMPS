import arsenal as ars
import numpy as np
import random
import os
import sys

## Description
# this script reads the output from a previous dnafold_lmp simulation and writes
  # the files necessary for restarting the simulation.
# if adding reserved staples, the restart geometry is used; otherwise, the binary
  # restart file is used.


################################################################################
### Parameters

def main():

	### input files
	simID = "16HB"
	simTag = ""
	srcFold = "/Users/dduke/Files/dnafold_lmp/"
	multiSim = False

	### parameters
	nstep = 1E6				# number of simulation steps
	addStap = True			# whether to add reserved staples
	r12_eq = 2.72			# equilibrium bead separation
	sigma = 2.14			# bead Van der Waals radius
	rseed = 1				# random seed

	### find simulation folder
	simFold  = srcFold + simID + simTag + "/"
	if multiSim:
		simFold += f"sim{rseed:02.0f}/"

	### add reserved staples
	if addStap:

		### read geometry file
		geoFile = simFold + "restart_geometry.out"
		points, strands, types, charges, bonds, angles = ars.readGeo(geoFile)
		dbox3 = ars.getDbox3(geoFile)
		nstrand = max(strands)

		### add reserved staples
		random.seed(rseed)
		stapFile = simFold + "metadata/reserved_staples.txt"
		is_reserved_strand = readRstap(stapFile, nstrand)
		points, types = addReservedStap(points, strands, types, is_reserved_strand, r12_eq, sigma, dbox3[0])
		bonds = updateBondTypes(bonds, types)

		### write geometry file
		outGeoFile = simFold + "restart_geometry.in"
		ars.writeGeo(outGeoFile, dbox3, points, strands, types, bonds, angles, nangleType=2, charges=charges, precision=16)

	### edit input file
	lammpsFile = simFold + "lammps.in"
	editInput(lammpsFile, nstep, addStap)


################################################################################
### File Handlers

### read old lammps file and write 
def editInput(lammpsFile, nstep, addStap):

	### read old lammps file
	ars.testFileExist(lammpsFile)
	with open(lammpsFile, 'r') as f:
		content_in = f.readlines()

	### parse the content and write edited content
	content_out = []
	i = 0
	while i < len(content_in):
		content_out.append(content_in[i])

		### set how to read geometry
		if content_in[i].startswith("## Geometry"):

			### if adding stales, use geometry
			if addStap:
				content_out.append("read_data       restart_geometry.in &\n")
				content_out.append("                extra/bond/per/atom 10 &\n")
				content_out.append("                extra/angle/per/atom 10 &\n")
				content_out.append("                extra/special/per/atom 100\n")

			### otherwise, use binary
			else:
				content_out.append("read_restart    restart_binary2.out\n")
				
			### skip the appropriate number of lines
			if "read_data" in content_in[i+1]:
				i += 4
			elif "read_restart" in content_in[i+1]:
				i += 1

		### remove scaffold relaxation
		elif content_in[i].startswith("## Relaxation"):
			content_out.pop()
			i += 8

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

		### next line
		i += 1

	### write edited lammps file
	with open(lammpsFile, 'w') as f:
		f.writelines(content_out)


### read reserved staple file
def readRstap(rstapFile, nstrand):
	is_reserved_strand = [ False for i in range(nstrand) ]

	### read reserved staples file
	ars.testFileExist(rstapFile,"reserved staples")
	with open(rstapFile, 'r') as f:
		reserved_strands = [ int(line.strip()) for line in f ]
	for si in range(len(reserved_strands)):
		is_reserved_strand[reserved_strands[si]-1] == True

	### return strand reservations status
	return is_reserved_strand


################################################################################
### Calculation Managers

### edit positions and types for adding reserved staples
def addReservedStap(r, strands, types, is_reserved_strand, r12_eq, sigma, dbox3):
	print("Initializing positions...")

	### parameters
	max_nfail_strand = 20
	max_nfail_bead = 20
	nbead = sum(strands>0)
	n_scaf = sum(strands==1)

	### initializations
	nbead_placed = n_scaf
	nbead_locked = n_scaf
	nstrand_locked = 1

	### loop over beads
	nfail_strand = 0
	while nbead_placed < nbead:
		bi = nbead_placed

		### skip if strand is already active
		if types[bi] != 3:

			### warning if trying to add active strand
			if is_reserved_strand[strands[bi]-1]:
				print("Flag: reserved staple already added, skipping.")

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
				r_propose = ars.applyPBC(r[bi-1] + r12_eq*ars.unitVector(ars.boxMuller()), dbox3)

			### random position for new strand
			else:
				r_propose = ars.randPos(dbox3)

			### evaluate position, break loop if no overlap
			if not ars.checkOverlap(r_propose,r[:bi],sigma,dbox3):
				break

			### if loop not broken update bead fail count
			nfail_bead += 1

			### break bead loop if too much failure
			if nfail_bead == max_nfail_bead:
				break

		### set position if all went well
		if nfail_bead < max_nfail_bead:
			r[bi] = r_propose
			types[bi] = 2
			nbead_placed += 1

			### update locked strands if end of strand
			if bi+1 == nbead or strands[bi] < strands[bi+1]:
				nbead_locked += sum(strands==nstrand_locked+1)
				nstrand_locked += 1
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