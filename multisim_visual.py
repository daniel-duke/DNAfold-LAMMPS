import arsenal as ars
from ovito import scene
from ovito.io import import_file
from ovito.modifiers import LoadTrajectoryModifier
from ovito.vis import Viewport, SimulationCellVis
import os

## Description
# this script loads the geometry and trajectory files from several simulation
  # "copies" (same simulation, different random seeds) and writes an ovito
  # session state file that visualizes all the trajectories at once.
# source folder must contain "copies.txt" file, which contains the names of the
  # directories containing the simulations to analyze.
# this script will only work if "backend_basics.py" has already been run for
  # the given simulation (requires a populated "analysis" folder).

## To Do
# figure out a way to set the default colors for numbered atom types (either
  # through python API or through default setting)


################################################################################
### Script

def main():

	### input files
	simID = "4HB"
	simTag = ""
	srcFold = "/Users/dduke/Files/dnafold_lmp/production/"
	datFileName = "trajectory_centered.dat"
	geoFileName = "geometry_vis.in"

	### output files
	ovitoFile = "folding_all.ovito"

	### get simulation folders
	simHomeFold = srcFold + simID + simTag + "/"
	simFolds, nsim = utils.getSimFolds(simHomeFold, True)

	### get base geometry
	geoFile = simFolds[0] + "analysis/" + geoFileName
	pipeline = import_file(geoFile, atom_style="molecular")
	pipeline.add_to_scene()

	### disable simulation cell
	vis_element = pipeline.source.data.cell.vis
	vis_element.enabled = False

	### adjust defaults
	particle_vis = pipeline.source.data.particles.vis
	particle_vis.radius = 1.2
	bonds_vis = pipeline.source.data.particles.bonds.vis
	bonds_vis.width = 2.4

	### set active viewport to top perspective
	viewport = scene.viewports.active_vp
	viewport.type = Viewport.Type.PERSPECTIVE
	viewport.camera_dir = (-1,0,0)
	viewport.camera_up = (0,1,0)
	viewport.zoom_all()

	### loop over simulations
	for i in range(nsim):

		### add trajectories to pipeline
		datFile = simFolds[i] + "analysis/" + datFileName
		traj_pipeline = import_file(datFile, multiple_frames=True)
		traj_mod = LoadTrajectoryModifier()
		traj_mod.source.load(datFile)
		pipeline.modifiers.append(traj_mod)

	### write ovito file
	ovitoFile = simHomeFold + "analysis/" + ovitoFile
	scene.save(ovitoFile)


### run the script
if __name__ == "__main__":
	main()

