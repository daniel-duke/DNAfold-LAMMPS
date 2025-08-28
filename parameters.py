import numpy as np
import sys

class parameters:

	### initialize
	def __init__( self, params):

		### set parameters
		self.rseed = params['rseed']
		self.rng = np.random.default_rng(params['rseed'])
		self.rseed_mis = params['rseed_mis']
		self.rng_mis = np.random.default_rng(params['rseed_mis'])
		self.nstep = params['nstep']
		self.nstep_relax = params['nstep_relax']
		self.dump_every = params['dump_every']
		self.dt = params['dt']
		self.dbox = params['dbox']
		self.stap_copies = params['stap_copies']
		self.circularScaf = params['circularScaf']
		self.reserveStap = params['reserveStap']
		self.forceBind = params['forceBind']
		self.startBound = params['startBound']
		self.nmisBond = params['nmisBond']
		self.ncompFactor = params['ncompFactor']
		self.optCompFactors = params['optCompFactors']
		self.optCompEfunc = params['optCompEfunc']
		self.bridgeEnds = params['bridgeEnds']
		self.dehyb = params['dehyb']
		self.debug = params['debug']
		self.T = params['T']
		self.T_relax = params['T_relax']
		self.gamma_t = 6*np.pi*params['visc']*params['r_h_bead']
		self.sigma = params['sigma']
		self.epsilon = 6.96*params['epsilon']
		self.r12_cut_WCA = params['sigma']*2**(1/6)
		self.r12_eq = params['r12_eq']
		self.k_x = 6.96*params['k_x']
		self.r12_cut_hyb = params['r12_cut_hyb']
		self.U_hyb = 6.96*params['U_hyb']
		self.U_mis_max = 6.96*params['U_mis_max']
		self.U_mis_min = 6.96*params['U_mis_min']
		self.U_mis_shift = 6.96*params['U_mis_shift']
		self.k_theta = params['dsLp']*0.0138*params['T']/params['r12_eq']
		self.n_scaf = 0
		self.n_stap = 0
		self.n_ori = 0
		self.nbead = 0
		self.nstrand = 0

		### check dump frequency
		if self.dump_every%2 != 0:
			print("Flag: Dump frequency must be an even number (for write restart purposes), adding 1.")
			self.dump_every += 1

		### check force binding and dehybridization
		if self.forceBind and self.dehyb:
			print("Flag: Dehybridization useless when force binding, removing dehybridization.")
			self.dehyb = False
		### check force binding and staple copies

		if self.forceBind and self.stap_copies > 1:
			print("Flag: Multiple staple copies useless when force binding, using only 1 staple copy.")
			self.forceBind = False

		### check force binding and misbinding
		if self.forceBind and self.nmisBound > 0:
			print("Flag: Misbinding useless when force binding, removing misbinding.")
			self.nmisBound = 0

		### check number fo complementary factors
		if self.nmisBond == 0 and self.ncompFactor > 1:
			print("Flag: Since no misbinding, setting number of complementary factors to 1.")
			self.ncompFactor = 1

		### check number fo complementary factors
		if self.nmisBond > 0 and self.ncompFactor <= 1:
			print("Flag: if including misbinding, number of complementary factors must be >1, setting to 2.")
			self.ncompFactor = 2

		### check number fo complementary factors
		if self.bridgeEnds == True:
			print("Flag: End bridging reactions do not work, proceed with caution.")
	

	### record values
	def record(self, paramsFile):
		with open(paramsFile,'w') as f:
			f.write(f"nstep             {self.nstep:0.0f}\n")
			f.write(f"nstep_relax       {self.nstep_relax:0.0f}\n")
			f.write(f"dump_every        {self.dump_every:0.0f}\n")
			f.write(f"dt [ns]           {self.dt}\n")
			f.write(f"dbox [nm]         {self.dbox:0.2f}\n")
			f.write(f"stap_copies       {self.stap_copies}\n")
			f.write(f"circularScaf      {self.circularScaf}\n")
			f.write(f"reserveStap       {self.reserveStap}\n")
			f.write(f"forceBind         {self.forceBind}\n")
			f.write(f"startBound        {self.startBound}\n")
			f.write(f"rseed_mis         {self.rseed_mis:0.0f}\n")
			f.write(f"nmisBond          {self.nmisBond}\n")
			f.write(f"ncompFactor       {self.ncompFactor}\n")
			f.write(f"optCompFactors    {self.optCompFactors}\n")
			f.write(f"optCompEfunc      {self.optCompEfunc}\n")
			f.write(f"dehyb             {self.dehyb}\n")
			f.write(f"T [K]             {self.T}\n")
			f.write(f"T_relax [K]       {self.T_relax}\n")
			f.write(f"U_hyb [kcal]      {self.U_hyb/6.96:0.2f}\n")

