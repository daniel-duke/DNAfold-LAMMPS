import armament as ars
import sys
import os

## Description
# this script contains functions that find DNAfold files on my local computer


################################################################################
### File Handlers

### return standard simulation location
def getSimHomeFold(desID, simTag="", sim_type="experiment"):
	return "/Users/dduke/Files/dnafold_lmp/" + sim_type + "/" + desID + simTag + "/"


### find the caDNAno file in my file system
def getCadFile(desID):

	### the root of all designs
	projectsFold = "/Users/dduke/Links/Projects/"
	cadFileName = desID + ".json"

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
def getOxFiles(desID, confTag):
	projectsFold = "/Users/dduke/Links/Projects/"
	oxFold = projectsFold + "dnafold_lmp/oxDNA_geometries/"
	topFile = oxFold + desID + ".top"
	confFile = oxFold + desID + confTag + ".dat"
	return topFile, confFile


### find the reserved staples file in my file system
def getRstapFile(desID, rstapTag):
	projectsFold = "/Users/dduke/Links/Projects/"
	rstapFold = projectsFold + "dnafold_lmp/reserved_staples/"
	rstapFile = rstapFold + "rstap_" + desID + rstapTag + ".txt"
	return rstapFile

