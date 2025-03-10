import arsenal as ars
from ovito import scene
from ovito.io import import_file
from ovito.modifiers import ComputePropertyModifier
from ovito.vis import Viewport, SimulationCellVis
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import utils

## Description
# this script reads either a caDNAno file or oxDNA configuration file to
  # get the positions for a DNA origami design, then it writes the lammps-
  # style geometry and ovito session state files for visualization.


################################################################################
### Parameters

def main():

	### input files
	simID = "16HB"
	simTag = ""
	srcFold = "/Users/dduke/Files/dnafold_lmp/"

	### analysis options
	position_src = "oxdna"		# where to get bead locations (cadnano or oxdna)
	reserveStap = False			# whether to remove reserved staples

	### get positions (using caDNAno)
	if position_src == "cadnano":
		cadFile = srcFold + simID + simTag + "/" + simID + ".json"
		r, strands = utils.initPositionsCaDNAno(cadFile)

	### get positions (using oxdna configuration)
	elif position_src == "oxdna":
		cadFile = srcFold + simID + simTag + "/" + simID + ".json"
		topFile = srcFold + simID + simTag + "/" + simID + ".top"
		confFile = srcFold + simID + simTag + "/" + simID + "_ideal.dat"
		r, strands = utils.initPositionsOxDNA(cadFile, topFile, confFile)

	### get reserved staples
	reserved_strands = []
	if reserveStap:
		rstapFile = srcFold + simID + simTag + "/reserved_staples.txt"
		reserved_strands = readRstap(rstapFile)

	### prepare the data for nice redering
	r, bonds, dbox3 = prepGeoData(r, strands, reserved_strands)

	### write geometry
	outFold = srcFold + simID + simTag + "/" + "analysis/"
	outGeoFile = outFold + "geometry_ideal.in"
	ars.createSafeFold(outFold)
	ars.writeGeo(outGeoFile, dbox3, r, types=strands, bonds=bonds)

	### write ovito file
	ovitoFile = outFold + "vis_ideal.ovito"
	writeOvito(ovitoFile, outGeoFile)


################################################################################
### File Handlers

### write session state vito file that visualizes the geometry
def writeOvito(ovitoFile, outGeoFile):

	### get base geometry
	pipeline = import_file(outGeoFile, atom_style="molecular")
	pipeline.add_to_scene()

	### disable simulation cell
	vis_element = pipeline.source.data.cell.vis
	vis_element.enabled = False

	### add compute properties
	pipeline.modifiers.append(ComputePropertyModifier(output_property='Radius',expressions=['(ParticleType==1)?0.6:1']))
	pipeline.modifiers.append(ComputePropertyModifier(operate_on='bonds',output_property='Width',expressions=['(BondType==1)?1.2:2']))

	### set active viewport to top perspective
	viewport = scene.viewports.active_vp
	viewport.type = Viewport.Type.PERSPECTIVE
	viewport.camera_dir = (-1,0,0)
	viewport.camera_up = (0,1,0)
	viewport.zoom_all()

	### write ovito file
	scene.save(ovitoFile)


################################################################################
### File Handlers

### read reserved staple file
def readRstap(rstapFile):

	### read reserved staples file
	ars.testFileExist(rstapFile,"reserved staples")
	with open(rstapFile, 'r') as f:
		reserved_strands = [ int(line.strip()) for line in f ]

	### return strand indices of reserved staples
	return reserved_strands


################################################################################
### Utility Functions

### positions and strands to 
def prepGeoData(r, strands, reserved_strands):
	n_ori = len(strands)
	n_scaf = strands.count(1)

	### box diameter
	dbox3 = [ max(abs(r[:,0]))+2.72, max(abs(r[:,1]))+2.4, max(abs(r[:,2]))+2.4 ]
	dbox3 = [ 2*i for i in dbox3 ]

	### neutralize reserved staples
	types = np.zeros(n_ori,dtype=int)
	for bi in range(n_ori):
		if strands[bi] in reserved_strands:
			r[bi] = [0,0,0]

	### get bonds
	bonds = np.zeros((0,3),dtype=int)
	for bi in range(n_ori-1):
		if strands[bi] == strands[bi+1]:
			bonds = np.append(bonds,[[strands[bi],bi+1,bi+2]],axis=0)

	### return results
	return r, bonds, dbox3


### run the script
if __name__ == "__main__":
	main()

