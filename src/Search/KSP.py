'''
KSP algorithm
'''

#!/usr/bin/python
import string
import argparse

# represents a node in the graph
class Node:
	def __init__(self, name):
		self.name = name	# name of the node
		self.dist = 1000000	# distance to this node from start node
		self.prev = None	# previous node to this node
		self.flag = 0		# access flag

# represents an edge in the graph
class Edge:
	def __init__(self, u, v, length):
		self.start = u
		self.end = v
		self.length = length

# read a text file and generate the graph according to declarations
def generateGraph(graph_file):
	V = []
	E = []
	fname = open(graph_file, "r")
	line = fname.readline()
	line = line[:-1]
	while line:
		taglist = line.split()
		if taglist[0] == 'node':
			V.append(Node(taglist[1]))
		elif taglist[0] == 'arc':
			E.append(Edge(taglist[1], taglist[2], float(taglist[3])))
		elif taglist[0] == 'edge':
			E.append(Edge(taglist[1], taglist[2], float(taglist[3])))
			E.append(Edge(taglist[2], taglist[1], float(taglist[3])))
		line = fname.readline()
		line = line[:-1]
	fname.close()
	return V, E

# generate the graph from a list of nodes and a list of edges
def generateGraphFromList(listNodes, listEdges, generateBackwardEdges=False):
	V = []
	E = []
	
	# generate the list of nodes
	for v in listNodes:
		V.append(Node(v))
	
	# generate the list of edges
	for e in listEdges:
		E.append(Edge(e[0], e[1], e[2]))
		# generate backward edges
		if generateBackwardEdges:
			E.append(Edge(e[1], e[0], e[2]))
	
	return V, E

# reset graph's variables to default
def resetGraph(N, E):
	for node in N:
		node.dist = 1000000.0
		node.prev = None
		node.flag = 0

# returns the smallest node in N but not in S
def pickSmallestNode(N):
	minNode = None
	for node in N:
		if node.flag == 0:
			minNode = node
			break
	if minNode == None:
		return minNode
	for node in N:
		if node.flag == 0 and node.dist < minNode.dist:
			minNode = node
	return minNode

# returns the list of edges starting in node u
def pickEdgesList(u, E):
	uv = []
	for edge in E:
		if edge.start == u.name:
			uv.append(edge)
	return uv

# returns the list of edges that start or end in node u
def pickEdgesListAll(u, E):
	uv = []
	for edge in E:
		if edge.start == u.name or edge.end == u.name:
			uv.append(edge)
	return uv

# Dijkstra's shortest path algorithm
def findShortestPath(N, E, origin, destination, ignoredEdges):
	
	# reset the graph (so as to discard information from previous runs)
	resetGraph(N, E)
	
	# set origin node distance to zero, and get destination node
	dest = None
	for node in N:
		if node.name == origin:
			node.dist = 0
		if node.name == destination:
			dest = node
	
	u = pickSmallestNode(N)
	while u != None:
		u.flag = 1
		uv = pickEdgesList(u, E)
		n = None
		for edge in uv:
			
			# avoid ignored edges
			if edge in ignoredEdges:
				continue
			
			# take the node n
			for node in N:
				if node.name == edge.end:
					n = node
					break
			if n.dist > u.dist + edge.length:
				n.dist = u.dist + edge.length
				n.prev = u
		
		u = pickSmallestNode(N)
		# stop when destination is reached
		if u == dest:
			break
	
	# generate the final path
	S = []
	u = dest
	while u.prev != None:
		S.insert(0,u)
		u = u.prev
	S.insert(0,u)
	
	return S

# print vertices and edges
def printGraph(N, E):
	print('vertices:')
	for node in N:
		previous = node.prev
		if previous == None:
			print(node.name, node.dist, previous)
		else:
			print(node.name, node.dist, previous.name)
	print('edges:')
	for edge in E:
		print(str(edge.start) + '|' + str(edge.end), edge.length)

# print S path
def printPath(S, E):
	strout = ''
	for node in S:
		if strout != '':
			strout += ' - '
		strout += node.name
		
	print(calcPathLength(S, E), "=", strout)

# generate a string from the path S in a specific format
def pathToString(S):
	strout = '['
	for i in range(0,len(S)-1):
		if i > 0:
			strout += ', '
		strout += '\'' + str(S[i].name) + '|' + str(S[i+1].name) + '\''
	return strout + ']'

# generate a list with the edges' names of a given route S
def pathToListOfString(S):
	lout = []
	for i in range(0,len(S)-1):
		lout.append(str(S[i].name) + '|' + str(S[i+1].name))
	return lout

# get the directed edge from u to v
def getEdge(E, u, v):
	for edge in E:
		if edge.start == u and edge.end == v:
			return edge
	return None

def runKShortestPathsStep(V, E, origin, destination, k, A, B):
	# Step 0: iteration 1
	if k == 1:
		A.append(findShortestPath(V, E, origin, destination, []))
		
	# Step I: iterations 2 to K
	else:
		lastPath = A[-1]        
		for i in range(0, len(lastPath)-1):
			# Step I(a)
			spurNode = lastPath[i]
			rootPath = lastPath[0:i+1]
			toIgnore = []
			
			for path in A:
				if path[0:i+1] == rootPath:
					ed = getEdge(E, spurNode.name, path[i+1].name)
					toIgnore.append(ed)
			
			# ignore the edges passing through nodes already in rootPath (except for the spurNode)
			for noder in rootPath[:-1]: 
				edgesn = pickEdgesListAll(noder, E)
				for ee in edgesn:
					toIgnore.append(ee)
			
			# Step I(b)
			spurPath = findShortestPath(V, E, spurNode.name, destination, toIgnore)
			if spurPath[0] != spurNode:
				continue
			
			# Step I(c)
			totalPath = rootPath + spurPath[1:]
			B.append(totalPath)
		
		# handle the case where no spurs (new paths) are available
		if not B:
			return False
			
		# Step II
		bestInB = None
		bestInBlength = 999999999
		for path in B:
			length = calcPathLength(path, E)
			if length < bestInBlength:
				bestInBlength = length
				bestInB = path
		A.append(bestInB)
		while bestInB in B:
			B.remove(bestInB)
		
	return True

# Yen's K shortest loopless paths algorithm
def KShortestPaths(V, E, origin, destination, K):
	# the K shortest paths
	A = []
	
	# potential shortest paths
	B = []
	
	for k in range(1,K+1):
		try:
 			if not runKShortestPathsStep(V, E, origin, destination, k, A, B):
			 	break
		except:
 			print('Problem on generating more paths@ Only ', k-1,' paths were found!')
 			break
		
	return A

# calculate path S's length
def calcPathLength(S, E):
	length = 0
	prev = None
	for node in S:
		if prev != None:
			length += getEdge(E, prev.name, node.name).length
		prev = node
	
	return length

# main procedure for many OD-pairs
def run(graph_file, OD_pairs, K):
	
	# read graph from file
	N, E = generateGraph(graph_file)
	
	#~ # find shortest path
	#~ S = findShortestPath(N, E, origin, destination, [])
	#~ printPath(S, E)
	
	#~ # find shortest path avoiding specific edges
	#~ S = findShortestPath(N, E, origin, destination, [E[1]])
	#~ printPath(S, E)
	
	# read list of OD-pairs
	OD = OD_pairs.split(';')
	for i in range(0,len(OD)):
		OD[i] = OD[i].split('|')
	
	# find K shortest paths of each OD-pair
	print('ksptable = [')
	lastod = len(OD)-1
	for iod, (o, d) in enumerate(OD):
		# find K shortest paths for this specific OD-pair
		S = KShortestPaths(N, E, o, d, K)
		
		# print the result for this specific OD-pair
		print('\t[ # ', str(o), '|', str(d),' flow')
		last = len(S)-1
		for i, path in enumerate(S):
			comma = ','
			if i == last:
				comma = ''
			print('\t\t', pathToString(path), comma, " # cost ", str(calcPathLength(path, E)))
		comma = ','
		if iod == lastod:
			comma = ''
		print('\t]', comma)
	print(']')

# return a list with the K shortest paths for the given origin-destination pair
# (this function was created to be called externally by another applications)
def getKRoutesNetFile(graph_file, origin, destination, K):
	
	lout = []
	
	# read graph from file
	N, E = generateGraph(graph_file)
	
	# find K shortest paths for this specific OD-pair
	S = KShortestPaths(N, E, origin, destination, K)
	
	for path in S:
		# store the path (in list of strings format) and cost to the out list 
		lout.append([pathToListOfString(path), calcPathLength(path, E)])
		
	return lout

# return a list with the K shortest paths for the given origin-destination pair,
# given the lists of nodes and edges (this function was created to be called 
# externally by another applications)
def getKRoutes(N, E, origin, destination, K):
	
	lout = []
	
	# find K shortest paths for this specific OD-pair
	S = KShortestPaths(N, E, origin, destination, K)
	
	for path in S:
		# store the path (in list of strings format) and cost to the out list 
		lout.append([pathToListOfString(path), calcPathLength(path, E)])
		
	return lout
	
# initializing procedure
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='KSP Algorithm',
		epilog='Please enter the correct format of the node/edge and weights',
		formatter_class=argparse.RawTextHelpFormatter)
	
	parser.add_argument('-f', dest='file', required=True,
						help='the graph file')
	parser.add_argument('-l', dest='OD_list', required=True,
						help='list of OD-pairs, in the format \'O|D\', where O are valid origin nodes, and D are valid destination nodes')
	parser.add_argument('-k', dest='K', type=int, required=True,
						help='number of shortest paths to find')
	args = parser.parse_args()
	
	graph_file = args.file
	OD_list = args.OD_list
	K = args.K
	
	run(graph_file, OD_list, K)
