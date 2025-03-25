import arsenal as ars
import sys
import os

## Description
# this script contains functions that find files on my local computer


################################################################################
### File Handlers

### find the caDNAno file in my file system
def getCadFile(simID):

	### the root of all designs
	projectsFold = "/Users/dduke/OneDrive - Duke University/DukeU/Research/Projects/"
	cadFileName = simID + ".json"

	### search dnafold_lmp caDNAno folder
	cadFold = projectsFold + "dnafold_lmp/cadnano/"
	for root, _, files in os.walk(cadFold):
		if cadFileName in files:
			return os.path.join(root, cadFileName)

	### search elementary caDNAno folder
	cadFold = projectsFold + "elementary/cadnano/"
	for root, _, files in os.walk(cadFold):
		if cadFileName in files:
			return os.path.join(root, cadFileName)

	### return error
	print("Unable to find caDNAno file, aborting.")
	sys.exit()


### find oxDNA geometry files, 
def getOxFiles(simID):
	projectsFold = "/Users/dduke/OneDrive - Duke University/DukeU/Research/Projects/"
	oxFold = projectsFold + "dnafold_lmp/oxDNA_geometries/"
	topFile = oxFold + simID + ".top"
	confFile = oxFold + simID + "_ideal.dat"
	return topFile, confFile


### find the reserved staples file in my file system
def getRstapFile(simID):
	projectsFold = "/Users/dduke/OneDrive - Duke University/DukeU/Research/Projects/"
	rstapFold = projectsFold + "dnafold_lmp/reserved_staples/"
	rstapFile = rstapFold + "rstap_" + simID + ".txt"
	return rstapFile

