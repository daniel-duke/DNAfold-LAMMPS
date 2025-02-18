import arsenal as ars
import numpy as np
import random
import shutil
import os
import sys

### write lammps-style geometry
def writeGeo(geoFile,dbox3,points,molecules="auto",types="auto",bonds="none",angles="none",natomType="auto",nbondType="auto",nangleType="auto",masses="auto",charges="none"):

	### notes
	# by convention, molecule/type/bond/angle indexing starts at 1, however, there is
	  # nothing in this function that requires this to be the case (it will print the
	  # values it is given).
	# assumes 4 decimal charge precision is sufficient. haha

	### count atoms
	natom = len(points)

	### interpret input
	if ars.isnumber(dbox3):
		dbox3 = [dbox3,dbox3,dbox3]
	elif not ars.isarray(dbox3) or len(dbox3) != 3:
		print("Flag: Not writing geometry file - dbox3 must be number or 3-element list.")
		return
	if isinstance(molecules,str) and molecules == "auto":
		molecules = np.zeros(natom,dtype=int)
	if isinstance(types,str) and types == "auto":
		types = np.ones(natom,dtype=int)
	if isinstance(bonds,str) and bonds == "none":
		bonds = np.zeros((0,3),dtype=int)
	if isinstance(angles,str) and angles == "none":
		angles = np.zeros((0,4),dtype=int)

	### inerpret charges
	if isinstance(charges,str) and charges == "none":
		is_charged = False
	elif isinstance(charges,str) and charges == "auto":
		is_charged = True
		charges = np.zeros(natom,dtype=int)
		len_charge = 1
	else:
		is_charged = True
		len_charge = len(str(int(max(charges))))

	### count objects
	nmolecule = int(max(molecules))
	nbond = len(bonds)
	nangle = len(angles)

	### some more input interpretation
	if isinstance(natomType,str) and natomType == "auto":
		natomType = int(max(types))
	if isinstance(nbondType,str) and nbondType == "auto":
		if nbond > 0:
			nbondType = int(max(bonds[:,0]))
		else:
			nbondType = 0
	if isinstance(nangleType,str) and nangleType == "auto":
		if nangle > 0:
			nangleType = int(max(angles[:,0]))
		else:
			nangleType = 0
	if isinstance(masses,str) and masses == "auto":
		masses = np.ones(natomType,dtype=int)

	### count digits
	len_natom = len(str(natom))
	len_nbond = len(str(nbond))
	len_nangle = len(str(nangle))
	len_nobject = max([len_natom,len_nbond,len_nangle])
	len_nmolecule = len(str(nmolecule))
	len_natomType = len(str(natomType))
	len_nbondType = len(str(nbondType))
	len_nangleType = len(str(nangleType))
	len_nobjectType = max([len_natomType,len_nbondType,len_nangleType])
	len_dbox3 = len(str(int(max(dbox3)/2)))

	### write to file
	with open(geoFile,'w') as f:

		f.write("## Number of Objects\n")
		f.write(f"\t{natom:<{len_nobject}} atoms\n")
		if nbond:
			f.write(f"\t{nbond:<{len_nobject}} bonds\n")
		if nangle:
			f.write(f"\t{nangle:<{len_nobject}} angles\n")
		f.write("\n")

		f.write("## Number of Object Types\n")
		f.write(f"\t{natomType:<{len_nobjectType}} atom types\n")
		if nbondType:
			f.write(f"\t{nbondType:<{len_nobjectType}} bond types\n")
		if nangleType:
			f.write(f"\t{nangleType:<{len_nobjectType}} angle types\n")
		f.write("\n")

		f.write("## Simulation Box\n")
		f.write(f"\t-{dbox3[0]/2:0{len_dbox3+3}.2f} {dbox3[0]/2:0{len_dbox3+3}.2f} xlo xhi\n")
		f.write(f"\t-{dbox3[1]/2:0{len_dbox3+3}.2f} {dbox3[1]/2:0{len_dbox3+3}.2f} ylo yhi\n")
		f.write(f"\t-{dbox3[2]/2:0{len_dbox3+3}.2f} {dbox3[2]/2:0{len_dbox3+3}.2f} zlo zhi\n\n")

		f.write("Masses\n\n")
		for i in range(natomType):
			f.write(f"\t{i+1:<{len_natomType}} {masses[i]}\n")
		f.write("\n")

		f.write("Atoms\n\n")
		for i in range(natom):
			f.write(f"\t{i+1:<{len_natom}} " + \
					f"{int(molecules[i]):<{len_nmolecule}} " + \
					f"{int(types[i]):<{len_natomType}} ")
			if is_charged:
				f.write(f"{charges[i]:>{len_charge+6}.4f}  ") 
			f.write(f"{points[i,0]:>{len_dbox3+4}.2f} " + \
					f"{points[i,1]:>{len_dbox3+4}.2f} " + \
					f"{points[i,2]:>{len_dbox3+4}.2f}\n")
		f.write("\n")

		if nbond:
			f.write("Bonds\n\n")
			for i in range(nbond):
				f.write(f"\t{i+1:<{len_nbond}} " + \
						f"{int(bonds[i,0]):<{len_nbondType}}  " + \
						f"{int(bonds[i,1]):<{len_natom}} " + \
						f"{int(bonds[i,2]):<{len_natom}}\n")
			f.write("\n")

		if nangle:
			f.write("Angles\n\n")
			for i in range(nangle):
				f.write(f"\t{i+1:<{len_nangle}} " + \
						f"{int(angles[i,0]):<{len_nangleType}}  " + \
						f"{int(angles[i,1]):<{len_natom}} " + \
						f"{int(angles[i,2]):<{len_natom}} " + \
						f"{int(angles[i,3]):<{len_natom}}\n")
			f.write("\n")
	return


### check for overlap between a bead and a list of beads
def checkOverlap(r0,r_other,sigma,dbox):
	for bi in range(len(r_other)):
		if np.linalg.norm(ars.applyPBC(r0-r_other[bi],dbox)) < sigma:
			return True
	return False


### return unit vector
def unitVector(vector):
	return vector / np.linalg.norm(vector)


### apply periodic boundary condition
def applyPBC(r,dbox):
	return r - dbox*np.round(r/dbox)


### vector with random orientation and (on average) unit magnitude
def boxMuller():
	x = np.zeros(3)
	x[0] = np.sqrt(-2 * np.log(random.uniform(0,1))) * np.cos(2 * np.pi * random.uniform(0,1))
	x[1] = np.sqrt(-2 * np.log(random.uniform(0,1))) * np.cos(2 * np.pi * random.uniform(0,1))
	x[2] = np.sqrt(-2 * np.log(random.uniform(0,1))) * np.cos(2 * np.pi * random.uniform(0,1))
	return x


### vector randomly placed within box
def randPos(dbox):
	x = np.zeros(3)
	x[0] = random.uniform(-dbox/2,dbox/2)
	x[1] = random.uniform(-dbox/2,dbox/2)
	x[2] = random.uniform(-dbox/2,dbox/2)
	return x


### test if files exist
def testFileExist(file,name):
	if os.path.isfile(file) == False:
		print("Error: Could not find " + name + " file:")
		print(file + "\n")
		sys.exit()


### creates new folder, only if it doesn't already exist
def createSafeFold(newFold):
	os.makedirs(newFold,exist_ok=True)


### creates new empty folder
def createEmptyFold(newFold):
	if os.path.exists(newFold):
		shutil.rmtree(newFold)
	os.makedirs(newFold)


### test if variable is numeric (both float and integer count)
def isnumber(x):
	try:
		value = float(x)
		return True
	except:
		return False


### check if variable is an array (both list and numpy array work)
def isarray(x):
	if isinstance(x,list):
		return True
	elif isinstance(x,np.ndarray):
		return True
	else:
		return False


