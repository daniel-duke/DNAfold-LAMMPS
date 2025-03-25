import arsenal as ars
import utils
import utilsLocal
from ovito import scene
from ovito.io import import_file
from ovito.modifiers import ComputePropertyModifier
from ovito.vis import Viewport, SimulationCellVis
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

## Description
# this script reads either a caDNAno file or oxDNA configuration file to
  # get the positions for a DNA origami design, then it writes the lammps-
  # style geometry and ovito session state files for visualization.


################################################################################
### Parameters

def main():

	### input files
	simID = "16HB"

	### analysis options
	position_src = "oxdna"		# where to get bead locations (cadnano or oxdna)
	includeStap = True			# whether to include staples
	reserveStap = False			# whether to remove reserved staples

	### output location
	simTag = ""
	srcFold = "/Users/dduke/Files/dnafold_lmp/"

	### get reserved staples
	reserved_strands = []
	if reserveStap:
		rstapFile = utilsLocal.getRstapFile(simID)
		reserved_strands = readRstap(rstapFile)

	### get positions from cadnano
	if position_src == "cadnano":
		cadFile = utilsLocal.getCadFile(simID)
		r, strands = utils.initPositionsCaDNAno(cadFile)

	### get positions from oxdna configuration
	elif position_src == "oxdna":
		cadFile = utilsLocal.getCadFile(simID)
		topFile, confFile = utilsLocal.getOxFiles(simID)
		r, strands = utils.initPositionsOxDNA(cadFile, topFile, confFile)

	### prepare the data for nice redering
	r, types, bonds, dbox3 = prepGeoData(r, strands, reserved_strands, includeStap)

	### write geometry
	outFold = srcFold + simID + simTag + "/" + "analysis/"
	outGeoFile = outFold + "geometry_ideal.in"
	ars.createSafeFold(outFold)
	ars.writeGeo(outGeoFile, dbox3, r, types=types, bonds=bonds)

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

	### set active viewport to top perspective
	viewport = scene.viewports.active_vp
	viewport.type = Viewport.Type.PERSPECTIVE
	viewport.camera_dir = (-1,0,0)
	viewport.camera_up = (0,1,0)
	viewport.zoom_all()

	### add compute properties
	pipeline.modifiers.append(ComputePropertyModifier(output_property='Radius',expressions=['(ParticleType==1)?0.6:1']))
	pipeline.modifiers.append(ComputePropertyModifier(operate_on='bonds',output_property='Width',expressions=['(BondType==1)?1.2:2']))

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

### get geometry data ready for visualization
def prepGeoData(r, strands, reserved_strands, includeStap):
	n_ori = len(strands)
	n_scaf = np.sum(strands==1)

	### remove staples
	if not includeStap:
		r = r[:n_scaf]
		strands = strands[:n_scaf]
		n_ori = n_scaf

	### remove reserved staples
	r_trim = np.zeros((0,3))
	strands_trim = np.zeros(0,dtype=int)
	for bi in range(n_ori):
		if strands[bi] not in reserved_strands:
			r_trim = np.append(r_trim,[r[bi,:]],axis=0)
			strands_trim = np.append(strands_trim,strands[bi])
	r = r_trim
	strands = strands_trim
	n_ori = len(strands)

	### get bonds
	bonds = np.zeros((0,3),dtype=int)
	for bi in range(n_ori-1):
		if strands[bi] == strands[bi+1]:
			bonds = np.append(bonds,[[strands[bi],bi+1,bi+2]],axis=0)

	### box diameter
	dbox3 = [ max(abs(r[:,0]))+2.72, max(abs(r[:,1]))+2.4, max(abs(r[:,2]))+2.4 ]
	dbox3 = [ 2*i for i in dbox3 ]

	### return results
	return r, strands, bonds, dbox3


### run the script
if __name__ == "__main__":
	main()

