import armament as ars
import utils
import utilsLocal
from ovito import scene
from ovito.io import import_file
from ovito.vis import Viewport, SimulationCellVis
from ovito.modifiers import ComputePropertyModifier
from ovito.modifiers import DeleteSelectedModifier
import argparse
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

	### where to get files
	useMyFiles = True

	### extract files from local system
	if useMyFiles:

		### chose design
		desID = "2HBx4"

		### analysis options
		confTag = None			# if using oxdna position, tag for configuration file
		rstapTag = None			# if reserving staples, tag for reserved staples file
		circularScaf = False	# whether to add bond between scaffold ends
		cornerView = False		# whether to view the origami at an angle

		### get input files
		cadFile = utilsLocal.getCadFile(desID)
		rstapFile = utilsLocal.getRstapFile(desID, rstapTag) if rstapTag is not None else None

		### determine position source
		if confTag is not None:
			position_src = "oxdna"
			topFile, confFile = utilsLocal.getOxFiles(desID, confTag)
		else:
			position_src = "cadnano"

		### set output folder
		outFold = utilsLocal.getSimHomeFold(desID)

	### use files in current folder
	if not useMyFiles:

		### get arguments
		parser = argparse.ArgumentParser()
		parser.add_argument('--cadFile',		type=str,	required=True,	help='name of caDNAno file')
		parser.add_argument('--topFile',		type=str, 	default=None,	help='if using oxdna positions, name of topology file')
		parser.add_argument('--confFile',		type=str, 	default=None,	help='if using oxdna positions, name of conformation file')
		parser.add_argument('--rstapFile',		type=str, 	default=None,	help='if reserving staples, name of reserved staples file')
		parser.add_argument('--circularScaf',	type=int,	default=True,	help='whether to add bond between scaffold ends')
		parser.add_argument('--cornerView',		type=int,	default=False,	help='whether to view the origami at an angle')

		### set arguments
		args = parser.parse_args()
		cadFile = args.cadFile
		topFile = args.topFile
		confFile = args.confFile
		rstapFile = args.rstapFile
		circularScaf = args.circularScaf
		cornerView = args.cornerView

		### determine position source
		if topFile is not None and confFile is not None:
			position_src = "oxdna"
		else:
			position_src = "cadnano"
			if topFile is not None:
				print("Flag: oxDNA topology file provided without configuration file, using caDNAno positions.")
			if confFile is not None:
				print("Flag: oxDNA configuration file provided without topology file, using caDNAno positions.")

		### set output folder
		outFold = "./"


################################################################################
### Heart

	### get reserved staples
	reserved_strands = []
	if rstapFile is not None:
		reserved_strands = readRstap(rstapFile)

	### get positions
	if position_src == "cadnano":
		r, strands = utils.initPositionsCaDNAno(cadFile)
	if position_src == "oxdna":
		print(cadFile)
		r, strands = utils.initPositionsOxDNA(cadFile, topFile, confFile)

	### prepare the data for nice redering
	types, bonds, dbox3 = prepGeoData(r, strands, reserved_strands, circularScaf)

	### write geometry
	ars.createSafeFold(outFold + "analysis")
	outGeoFile = outFold + "analysis/geometry_ideal.in"
	ars.writeGeo(outGeoFile, dbox3, r, types=types, bonds=bonds)

	### write ovito file
	ovitoFile = outFold + "analysis/vis_ideal.ovito"
	writeOvito(ovitoFile, outGeoFile, cornerView)


################################################################################
### File Handlers

### write session state vito file that visualizes the geometry
def writeOvito(ovitoFile, outGeoFile, cornerView):

	### set colors
	scaf_color = ars.getColor("orchid")
	stap_color = ars.getColor("silver")

	### get base geometry
	pipeline = import_file(outGeoFile, atom_style="molecular")
	pipeline.add_to_scene()

	### prepare basic DNAfold scene
	pipeline = utils.setOvitoBasics(pipeline)

	### set active viewport to corner view
	if cornerView:
		viewport = scene.viewports.active_vp
		viewport.camera_dir = (1,-1,-1)
		viewport.camera_up = (0,0,1)
		viewport.zoom_all()

	### set scaffold and staple particle radii and bond widths
	pipeline.modifiers.append(ComputePropertyModifier(output_property='Radius',expressions=['(ParticleType==1)?0.6:1']))
	pipeline.modifiers.append(ComputePropertyModifier(operate_on='bonds',output_property='Width',expressions=['(BondType==1)?1.2:2']))

	### set colors
	pipeline.modifiers.append(ComputePropertyModifier(output_property='Color',expressions=[f'(ParticleType==1)?{scaf_color[0]}/255:{stap_color[0]}/255',f'(ParticleType==1)?{scaf_color[1]}/255:{stap_color[1]}/255',f'(ParticleType==1)?{scaf_color[2]}/255:{stap_color[2]}/255']))

	### remove reserved staples, or all staples
	pipeline.modifiers.append(ComputePropertyModifier(enabled=False,output_property='Selection',expressions=['ParticleType==3']))
	pipeline.modifiers.append(ComputePropertyModifier(enabled=False,output_property='Selection',expressions=['ParticleType!=1']))
	pipeline.modifiers.append(DeleteSelectedModifier())

	### write ovito file
	scene.save(ovitoFile)
	pipeline.remove_from_scene()


################################################################################
### File Handlers

### read staple file
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
def prepGeoData(r, strands, reserved_strands, circularScaf):
	n_ori = len(strands)
	n_scaf = np.sum(strands==1)
	n_padCell = 2

	### types
	types = np.ones(n_ori)
	for bi in range(n_scaf,n_ori):
		if strands[bi] in reserved_strands:
			types[bi] = 3
		else:
			types[bi] = 2

	### get bonds
	bonds = np.zeros((0,3),dtype=int)
	for bi in range(n_ori-1):
		if strands[bi] == strands[bi+1]:
			bonds = np.append(bonds,[[strands[bi],bi+1,bi+2]],axis=0)

	### add scaffold ends bond
	if circularScaf:
		bonds = np.append(bonds, [[1,1,n_scaf]], axis=0)

	### box diameter
	dbox3 = [ max(abs(r[:,0]))+n_padCell*2.72, max(abs(r[:,1]))+n_padCell*2.4, max(abs(r[:,2]))+n_padCell*2.4 ]
	dbox3 = [ 2*i for i in dbox3 ]

	### return results
	return types, bonds, dbox3


### run the script
if __name__ == "__main__":
	main()
	print()

