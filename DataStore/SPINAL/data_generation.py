import sys
import networkx as nx

class GenerateData:
	def __init__(self, ppi1filename, ppi2filename, simfilename):
                outppi1filename = "%s.gml" % ppi1filename
		outppi2filename = "%s.gml" % ppi2filename
		outsimfilename = "%s.pin" % simfilename

		ppi1file = open(ppi1filename)
		ppi2file = open(ppi2filename)
		self.ppi1 = nx.Graph()
		self.ppi2 = nx.Graph()
		for i, gt in enumerate(([self.ppi1, ppi1file], [self.ppi2, ppi2file])):
			gt[1].readline()
			for line in gt[1]:
				cols = line.split()
				gt[0].add_edge(cols[0], cols[1])
			for j, protein in enumerate(gt[0].nodes()):
				gt[0].node[protein]['index'] = j
		self.ppi1.remove_edges_from(self.ppi1.selfloop_edges())
		self.ppi2.remove_edges_from(self.ppi2.selfloop_edges())

		simfile = open(simfilename)
		outsimfile = open(outsimfilename, "w")
		max_sim = 0.0
		for line in simfile:
			cols = line.split()
			max_sim = max(max_sim, float(cols[2]))
		simfile.close()
		simfile = open(simfilename)

		for line in simfile:
			cols = line.split()
			n1 = cols[0]
			n2 = cols[1]
			nokey = False
			try:
				in1 = self.ppi1.node[n1]['index']
			except KeyError:
				nokey = True
			try:
				in2 = self.ppi2.node[n2]['index']
			except KeyError:
				nokey = True
			if nokey == False:
				sim_value = float(cols[2]) / max_sim
				line = "%s %s %s\n" %(in1, in2, sim_value)
				outsimfile.write(line)
		nx.write_gml(self.ppi1, outppi1filename)
		nx.write_gml(self.ppi2, outppi2filename)
		simfile.close()
		outsimfile.close()
		ppi1file.close()
		ppi2file.close()

	
if __name__ == "__main__":				
	inputdata = GenerateData(sys.argv[1], sys.argv[2], sys.argv[3])		
																					
