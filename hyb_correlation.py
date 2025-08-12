import armament as ars
import utils
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from ovito import scene
from ovito.io import import_file
from ovito.vis import Viewport
from ovito.modifiers import ComputePropertyModifier
from ovito.modifiers import ColorCodingModifier
from ovito.modifiers import DeleteSelectedModifier
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pickle
import sys
import os

## Description
# this script reads a bath of DNAfold trajectories and calculates the correlation
  # of the staple hybridization times with some measure of the final product
  # quality (for now, either crystallinity or final number of bound staples).
# this script will only work if "backend_basics.py" has already been run for
  # the given simulation (requires a populated "analysis" folder).
# the hybridization / product quality correlation (hybCorr) is a rather simple
  # concept when applied to individual beads; however, when extracting a
  # single correlation value for each strand, there are several options.
  # First, there are the options where some "composite" hyb time is calculed
  # for each strand, then this value is correlated with the product quality.
  # This composite value can be either the average, earliest, or latest of
  # the hyb times of the component beads. Second, there are the options where
  # the correlation is calulated for each component bead, then some
  # "composite" correlation is calculated. This composite value can be either 
  # the average or the max (absolute value) correlation.


################################################################################
### Parameters

def main():

	### get arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--cadFile',		type=str,	required=True,		help='name of caDNAno file')
	parser.add_argument('--copiesFile',		type=str,	required=True,		help='name of copies file, which contains a list of simulation folders')	
	parser.add_argument('--topFile',		type=str, 	default=None,		help='if using oxdna positions, name of topology file')
	parser.add_argument('--confFile',		type=str, 	default=None,		help='if using oxdna positions, name of conformation file')
	parser.add_argument('--corr_var',		type=str,	default="final_S",	help='what to correlate with the strands (values: final_S, final_n_hyb)')
	parser.add_argument('--corr_type',		type=str,	default="hyb_last",	help='how to average correlation across strand (values: hyb_avg, hyb_first, hyb_last, corr_avg, corr_max')
	parser.add_argument('--values_report',	type=str,	default="none",		help='what staple results to report (values: none, staple, corr, all')
	parser.add_argument('--loadResults',	type=int,	default=False,		help='whether to load the results from a pickle file')
	parser.add_argument('--nstep_skip',		type=float,	default=0,			help='if not loading results, number of recorded initial steps to skip')
	parser.add_argument('--nstep_max',		type=float,	default=0,			help='if not loading results, max number of recorded steps to use (0 for all)')
	parser.add_argument('--coarse_time',	type=int,	default=1,			help='if not loading results, coarse factor for time steps')
	parser.add_argument('--mov_avg_stride',	type=int,	default=1,			help='stride length for moving average')

	### set arguments
	args = parser.parse_args()
	cadFile = args.cadFile
	copiesFile = args.copiesFile
	topFile = args.topFile
	confFile = args.confFile
	corr_var = args.corr_var
	corr_type = args.corr_type
	values_report = args.values_report
	loadResults = args.loadResults
	nstep_skip = int(args.nstep_skip)
	nstep_max = int(args.nstep_max)
	coarse_time = args.coarse_time
	mov_avg_stride = args.mov_avg_stride

	### check input
	if topFile is not None and confFile is not None:
		position_src = "oxdna"
	else:
		position_src = "cadnano"
		if topFile is not None:
			print("Flag: oxDNA topology file provided without configuration file, using caDNAno positions.")
		if confFile is not None:
			print("Flag: oxDNA configuration file provided without topology file, using caDNAno positions.")


################################################################################
### Heart

	### read data and calculate results
	if not loadResults:

		### get simulation folders
		simFolds, nsim = utils.getSimFolds(copiesFile)

		### get pickled data
		connFile = "analysis/connectivity_vars.pkl"
		strands, bonds_backbone, complements, n_scaf, nbead = readConn(connFile)

		### get minimum number of steps
		nstep_allSim = np.zeros(nsim,dtype=int)
		for i in range(nsim):
			datFile = simFolds[i] + "analysis/trajectory_centered.dat"
			nstep_allSim[i] = ars.getNstep(datFile, nstep_skip, coarse_time)
		nstep_min = int(min(nstep_allSim))
		nstep_use = nstep_min if nstep_max == 0 else min([nstep_min,nstep_max])

		### loop through simulations
		first_hyb_times_scaled_allSim = np.zeros((nsim,n_scaf))
		n_hyb_frac_allSim = np.zeros((nsim,nstep_use))
		S_allSim = np.zeros((nsim,nstep_use))
		for i in range(nsim):

			### calculate hyb times
			hybFile = simFolds[i] + "analysis/hyb_status.dat"
			hyb_status = utils.readHybStatus(hybFile, nstep_skip, coarse_time, nstep_use)
			dump_every = utils.getDumpEveryHyb(hybFile)*coarse_time
			first_hyb_times_scaled_allSim[i] = utils.calcFirstHybTimes(hyb_status, complements, n_scaf, dump_every)[1]
			n_hyb_frac_allSim[i] = np.sum(hyb_status==1,axis=1)/nbead

			### calculate crystallinity
			datFile = simFolds[i] + "analysis/trajectory_centered.dat"
			bdis_scaf = list(range(1,n_scaf+1))
			points, _, dbox = ars.readAtomDump(datFile, nstep_skip, coarse_time, bdis=bdis_scaf, nstep_max=nstep_use); print()
			S_allSim[i] = utils.calcCrystallinity(points, dbox)

		### store results
		resultsFile = "analysis/hyb_correlation_results.pkl"
		with open(resultsFile, 'wb') as f:
			pickle.dump([first_hyb_times_scaled_allSim, n_hyb_frac_allSim, S_allSim, strands, bonds_backbone, complements, n_scaf], f)

	### load results
	else:
		resultsFile = "analysis/hyb_correlation_results.pkl"
		ars.testFileExist(resultsFile,"results")
		with open(resultsFile, 'rb') as f:
			[first_hyb_times_scaled_allSim, n_hyb_frac_allSim, S_allSim, strands, bonds_backbone, complements, n_scaf] = pickle.load(f)


################################################################################
### Results

	### apply moving average
	for i in range(len(S_allSim)):
		S_allSim[i] = ars.movingAvg(S_allSim[i], mov_avg_stride)

	### select product quality variable for hybridization correlation
	if corr_var == "final_n_hyb":
		product_quality = n_hyb_frac_allSim[:,-1]
	elif corr_var == "final_S":
		product_quality =  S_allSim[:,-1]
	else:
		print("Error: Unknown correlation variable.")
		sys.exit()

	### calculate and visualize correlation
	hybCorr, hybCorr_strand, h_strand = calcHybCorr(first_hyb_times_scaled_allSim, product_quality, strands, complements, corr_type)
	
	### machine learning
	linReg(h_strand,product_quality)
	lasso(h_strand,product_quality)
	linRegPCA(h_strand,product_quality)
	lassoPCA(h_strand,product_quality)
	mlp(h_strand,product_quality)
	# plt.show()

	### prepare position data
	if position_src == "cadnano":
		r = utils.initPositionsCaDNAno(cadFile)[0]
	if position_src == "oxdna":
		r = utils.initPositionsOxDNA(cadFile, topFile, confFile)[0]
	r, charges, dbox3 = prepGeoData(r, strands, complements, hybCorr, hybCorr_strand)

	### write geometry and ovito files
	outGeoFile = "analysis/geometry_hybCorr.in"
	ars.writeGeo(outGeoFile, dbox3, r, types=strands, charges=charges, bonds=bonds_backbone)
	ovitoFile = "analysis/vis_hybCorr.ovito"
	writeOvito(ovitoFile, outGeoFile)

	### report best staples
	staple_labels = np.arange(2,max(strands)+1)
	sorted_results = sorted(zip(hybCorr_strand[1:],staple_labels), reverse=True)
	for corr, s in sorted_results:
		if values_report == "staple":
			print(f"{s:2.0f}")
		if values_report == "corr":
			print(f"{corr:0.4f}")
		if values_report == "all":
			print(f"Strand {s:2.0f}: {corr:0.4f}")

	### correlation statistics
	plotCorrHist(first_hyb_times_scaled_allSim, product_quality, strands, complements, corr_type)
	# plt.show()


################################################################################
### Plotters

### plot histogram of hybridization correlations
def plotCorrHist(first_hyb_times_scaled_allSim, product_quality, strands, complements, corr_type):

	### actual correlation
	hybCorr_strand = calcHybCorr(first_hyb_times_scaled_allSim, product_quality, strands, complements, corr_type)[1]

	### random correlation
	nrand = 100
	nstrand = max(strands)
	hybCorr_strand_rand = np.zeros((nrand,nstrand))
	for i in range(nrand):
		np.random.shuffle(product_quality)
		hybCorr_strand_rand[i] = calcHybCorr(first_hyb_times_scaled_allSim, product_quality, strands, complements, corr_type)[1]
	hybCorr_strand_rand = hybCorr_strand_rand.reshape(nrand*nstrand)

	### plot
	nbin = 10
	Alim = [-1,1]
	ars.magicPlot()
	ars.plotHist(hybCorr_strand_rand,nbin=nbin*2,Alim_bin=Alim,plotAsLine=True)
	ars.plotHist(hybCorr_strand,nbin=nbin,Alim_bin=Alim)
	plt.legend(["Randomized","Original Data"])


### write session state vito file that visualizes the geometry
def writeOvito(ovitoFile, outGeoFile):

	### initialize pipeline
	pipeline = import_file(outGeoFile, atom_style="full")
	pipeline.add_to_scene()

	### prepare basic DNAfold scene
	pipeline = utils.setOvitoBasics(pipeline)

	### set scaffold and staple particle radii and bond widths
	pipeline.modifiers.append(ComputePropertyModifier(output_property='Radius',expressions=['(ParticleType==1)?0.6:1']))
	pipeline.modifiers.append(ComputePropertyModifier(operate_on='bonds',output_property='Width',expressions=['(@1.ParticleType==1)?1.2:2']))

	### set color coding
	pipeline.modifiers.append(ColorCodingModifier(property='Charge',start_value=-1,end_value=1,gradient=ColorCodingModifier.BlueWhiteRed()))
	pipeline.modifiers.append(ComputePropertyModifier(output_property='Color',expressions=['(ParticleType==1)?1:Color.R','(ParticleType==1)?1:Color.G','(ParticleType==1)?1:Color.B']))

	### add option to delete staples
	pipeline.modifiers.append(ComputePropertyModifier(enabled=False,output_property='Selection',expressions=['ParticleType!=1']))
	pipeline.modifiers.append(DeleteSelectedModifier())

	### write ovito file
	scene.save(ovitoFile)
	pipeline.remove_from_scene()


################################################################################
### File Handlers

def readConn(connFile):
	ars.testFileExist(connFile, "connectivity")
	with open(connFile, 'rb') as f:
		params = pickle.load(f)[0]
	strands = params['strands']
	bonds_backbone = params['bonds_backbone']
	complements = params['complements']
	n_scaf = params['n_scaf']
	nbead = params['nbead']
	return strands, bonds_backbone, complements, n_scaf, nbead


################################################################################
### Calculation Managers

### calculate correlation between hybridization times (h) and product quality (q)
def calcHybCorr(h, q, strands, complements, corr_type):
	nsim = h.shape[0]
	n_scaf = h.shape[1]
	nstrand = max(strands)
	h_strand = np.zeros((nsim,nstrand))
	corr_strand = np.zeros(nstrand)

	### correlation by bead
	corr = np.zeros(n_scaf)
	for bi in range(n_scaf):
		if len(set(h[:,bi])) != 1:
			corr[bi] = np.corrcoef(h[:,bi],q)[0,1]

	### group correlation by strands, pre-correlation methods
	h_grouped = [ np.zeros((nsim,0)) for i in range(nstrand) ]
	h_grouped[0] = np.zeros((nsim,1))
	for bi in range(n_scaf):
		if len(complements[bi]) > 0:
			strand = strands[complements[bi][0]-1]
			A = h_grouped[strand-1]
			B = h[:,bi].reshape(nsim,1)
			h_grouped[strand-1] = np.concatenate((A,B),axis=1)
	for si in range(1,nstrand):
		if corr_type == "hyb_avg":
			h_strand[:,si] = np.mean(h_grouped[si],axis=1)
		elif corr_type == "hyb_first":
			h_strand[:,si] = np.min(h_grouped[si],axis=1)
		elif corr_type == "hyb_last":
			h_strand[:,si] = np.max(h_grouped[si],axis=1)
		if corr_type == "hyb_avg" or corr_type == "hyb_first" or corr_type == "hyb_last":
			if len(set(h_strand[:,si])) != 1:
				corr_strand[si] = np.corrcoef(h_strand[:,si],q)[0,1]

	### group correlation by strands, post-correlation methods
	if corr_type == "corr_avg" or corr_type == "corr_max":
		corr_grouped = [ [] for si in range(nstrand) ]
		corr_grouped[0] = [0]
		for bi in range(n_scaf):
			if len(complements[bi]) > 0:
				strand = strands[complements[bi][0]-1]
				corr_grouped[strand-1].append(corr[bi])
		if corr_type == "corr_avg":
			corr_strand = np.array([ np.mean(corr_grouped[si]) for si in range(nstrand) ])
		elif corr_type == "corr_max":
			for si in range(nstrand):
				if all(c > 0 for c in corr_grouped[si]):
					corr_strand[si] = max(corr_grouped[si])
				elif all(c < 0 for c in corr_grouped[si]):
					corr_strand[si] = min(corr_grouped[si])
				else:
					corr_strand[si] = 0

	### error message
	elif corr_type != "hyb_avg" and corr_type != "hyb_first" and corr_type != "hyb_last":
		print("Error: Unknown correlation type.")
		sys.exit()

	### results
	return corr, corr_strand, h_strand


### boring
def linReg(X, y):

	# Split into training and test sets (80% train, 20% test)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Fit Linear Regression model
	model = LinearRegression()
	model.fit(X_train, y_train)

	# Predict and evaluate
	y_pred = model.predict(X_test)
	mse = mean_squared_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)

	print(f"Mean Squared Error (MSE): {mse:0.4f}")
	print(f"R-squared (R²): {r2:0.4f}")

	# Optional: plot predicted vs actual
	plt.figure("LR", figsize=(8,6))
	plt.scatter(y_test, y_pred)
	plt.xlabel("True values")
	plt.ylabel("Predicted values")
	plt.title("Linear Regression: Test Set Predictions")
	plt.grid(True)


### giddy up
def lasso(X, y):
	# Split into training and test sets (80% train, 20% test)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Fit Lasso Regression model
	lasso = Lasso(alpha=0.1)
	lasso.fit(X_train, y_train)

	# Predict and evaluate
	y_pred = lasso.predict(X_test)
	mse = mean_squared_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)

	print(f"Mean Squared Error (MSE): {mse:0.4f}")
	print(f"R-squared (R²): {r2:0.4f}")

	# Optional: plot predicted vs actual
	plt.figure("Lasso", figsize=(8,6))
	plt.scatter(y_test, y_pred)
	plt.xlabel("True values")
	plt.ylabel("Predicted values")
	plt.title("Lasso Regression: Test Set Predictions")
	plt.grid(True)


### less boring
def linRegPCA(X, y):

	# Train-test split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Apply PCA (retain 95% variance)
	pca = PCA(n_components=0.95)
	X_train_pca = pca.fit_transform(X_train)
	X_test_pca = pca.transform(X_test)

	# Fit Linear Regression on PCA-transformed data
	model = LinearRegression()
	model.fit(X_train_pca, y_train)

	# Predict and evaluate
	y_pred = model.predict(X_test_pca)
	mse = mean_squared_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)

	print(f"Mean Squared Error (MSE): {mse:0.4f}")
	print(f"R-squared (R²): {r2:0.4f}")
	print(f"Number of PCA components used: {pca.n_components_}")

	# Optional: plot predictions
	plt.figure("LR PCA", figsize=(8,6))
	plt.scatter(y_test, y_pred)
	plt.xlabel("True values")
	plt.ylabel("Predicted values")
	plt.title("Linear Regression with PCA: Test Set Predictions")
	plt.grid(True)


### best bet
def lassoPCA(X, y):

	# Train-test split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Apply PCA (retain 95% of variance or manually pick n_components)
	pca = PCA(n_components=0.95)
	X_train_pca = pca.fit_transform(X_train)
	X_test_pca = pca.transform(X_test)

	# Lasso with cross-validation to select best alpha
	lasso = LassoCV(cv=5, random_state=42).fit(X_train_pca, y_train)

	# Predict and evaluate
	y_pred = lasso.predict(X_test_pca)
	mse = mean_squared_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)

	print(f"Mean Squared Error (MSE): {mse:0.4f}")
	print(f"R-squared (R²): {r2:0.4f}")
	print(f"Number of PCA components used: {pca.n_components_}")
	print(f"Best alpha: {lasso.alpha_:0.4f}")

	# Plot true vs predicted
	plt.figure("Lasso PCA", figsize=(8,6))
	plt.scatter(y_test, y_pred)
	plt.xlabel("True values")
	plt.ylabel("Predicted values")
	plt.title("Lasso Regression with PCA: Test Set Predictions")
	plt.grid(True)


### bestest bet
def mlp(X, y):

	# Train-test split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Train MLP
	mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=1000, random_state=42)
	mlp.fit(X_train, y_train)

	# Predict and evaluate
	y_pred = mlp.predict(X_test)
	mse = mean_squared_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)

	print(f"MSE: {mse:0.4f}")
	print(f"R²: {r2:0.4f}")

	# Plot true vs predicted
	plt.figure("MLP", figsize=(8,6))
	plt.scatter(y_test, y_pred)
	plt.xlabel("True Values")
	plt.ylabel("Predicted Values")
	plt.title("MLP Regression: Test Set Predictions")
	plt.grid(True)


################################################################################
### Utility Functions

### get geometry data ready for visualization
def prepGeoData(r, strands, complements, hybCorr, hybCorr_strand):
	n_ori = len(strands)
	n_scaf = np.sum(strands==1)

	### box diameter
	dbox3 = [ max(abs(r[:,0]))+2.72, max(abs(r[:,1]))+2.4, max(abs(r[:,2]))+2.4 ]
	dbox3 = [ 2*i for i in dbox3 ]

	### assign charge
	charges = np.zeros(n_ori)
	charges[:n_scaf] = hybCorr
	for bi in range(n_scaf,n_ori):
		charges[bi] = hybCorr_strand[strands[bi]-1]

	### return results
	return r, charges, dbox3


### run the script
if __name__ == "__main__":
	main()
	print()

