"""
this module contains all the constant variables used
"""

import os

# constant paths
BASE_PATH = os.path.abspath(os.path.dirname((__file__)))
DATA_PATH = os.path.join(BASE_PATH, 'DataStore')
BLAST_PATH = os.path.join(DATA_PATH, 'BLAST')
STRING_PATH = os.path.join(DATA_PATH, 'string_db')
JSON_PATH = os.path.join(DATA_PATH, 'jsons')
OBJ_PATH = os.path.join(DATA_PATH, 'objects')
NP_PATH = os.path.join(DATA_PATH, 'numpy')
ISON_PATH = os.path.join(DATA_PATH, 'isorankN')
NETAL_PATH = os.path.join(DATA_PATH, 'NETAL')
PINALOG_PATH = os.path.join(DATA_PATH, 'pinalog')
GRAAL_PATH = os.path.join(DATA_PATH, 'GRAAL')
CGRAAL_PATH = os.path.join(DATA_PATH, 'C-GRAAL')
MIGRAAL_PATH = os.path.join(DATA_PATH, 'MI-GRAAL')
HUBALIGN_PATH = os.path.join(DATA_PATH, 'HubAlign')
MAGNA_PATH = os.path.join(DATA_PATH, 'MAGNA')
PROPER_PATH = os.path.join(DATA_PATH, 'PROPER')
SPINAL_PATH = os.path.join(DATA_PATH, 'SPINAL')
SVG_PATH = os.path.join(DATA_PATH, 'figures')
GEPHI_PATH = os.path.join(DATA_PATH, 'figures')
GO_FILE = os.path.join(STRING_PATH, 'all_go_knowledge_full.tsv')
REACTOME_PATH = os.path.join(DATA_PATH, 'reactome')
PW_FILE = os.path.join(REACTOME_PATH, 'UniProt2Reactome_All_Levels.txt')
OPTNET_PATH = os.path.join(DATA_PATH, 'optnet')
MODULEALIGN_PATH = os.path.join(DATA_PATH, 'moduleAlign')

#CLOUD constant
CHECK_FILES = False

# parse constants
INTERACTION_THR = 900  # threshold for propper interaction score in string_db
GO_REPORT_FREQ = 1000

# power method constants
POWER_METHOD_REPORT_FREQ = 10**7  # report frequency for power method progress
POWER_METHOD_ERROR_THR = 10**-6  # maximum squared error for power method
MIN_POWER_METHOD_ITERS = 3  # minimum iterations for power method to finish
MAX_POWER_METHOD_ITERS = 20  # maximum iterations for power method to finish
ALPHA_BIAS = 0.6  # power method alpha bias for each iteration

# clustering constants
CLUSTERS_COUNT = 40  # for noisy spectral clustering
NOISE_STRENGTH = 0.7  # noise range to be added for noisy spectral clustering
MAX_CLUSTER_SIZE = 500  # preffered maximum cluster size

# seed extend constants
TOPO_STRENGTH = 0.9  # the ratio to use for topology sim and base sim 0.9
SEED_KEEP_RATIO = 0.3  # keep some and leave some
SEED_PR_ALPHA = 0.85  # Damping parameter for PageRank
MAX_SEED_SIZE = 1400  # base seed limit
EXTEND_KEEP_RATIO = 0.01  # keep some and leave some
MAX_EXTEND_SIZE = 3000  # extend seed limit
BAD_EDGE_COST = 0.0001  # bad back edge cost
BLAST_CUT = 150

# matching coefs
BLAST_COEF = 1.0
DEGREE_COEF = 0.0
SEED_FACTOR_COEF = 0.0
BLAST_TH = 3e-5

#seed extend SA constants
T_HIGH = 10 ** 1
T_LOW = 9.9
SWAP_SIZE = 2

# extra for neighbor seed extend
NEIGHBOR_STRENGTH = 0

# munkres constants
MUNKRES_RANDOM_NOISE = 0

# isorankN constants
# DATA_INPUT = 'data.inp'
ISON_ALPHA = 0.6
ISON_ITERS = 10
ISON_THRESH = 1e-3
ISON_MAXVL = 1e4

# NETAL constants
NETAL_AA = 0.0001
NETAL_BB = 0
NETAL_CC = 1
NETAL_IT = 10

# MAGNA constants
MAGNA_P = 1000
MAGNA_N = 15000
MAGNA_ALPHA = 0.7

# PROPER constants
PROPER_R = 1
PROPER_L = 150

# SPINAL constants
SPINAL_ALG = 'I'
SPINAL_ALPHA = 0.7

# optnet constants
optnet_alg = 's3'  # choices are: s3, ics, ec, s3denom, icstimesec, s3variant
optnet_cxrate = '0.05'
optnet_cxswappb = '0.75'
optnet_mutrate = '0.05'
optnet_mutswappb = '0.0001'
optnet_oneobjrate = '0.75'
optnet_popsize = '200'
optnet_generations = '1000000000'
optnet_hillclimbiters = '10000'
optnet_timelimit = '20'  # minutes

# moduleAlign constants
moduleAlign_alpha = '0.3'

# visualization constants
NORM_MARGIN = 0.05
MIN_VIS_CUT = 0.00001

# pathway score constants
PWS2_LIMIT = 2
