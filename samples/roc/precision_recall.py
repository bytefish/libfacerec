import sys
import numpy as np
import matplotlib.pyplot as plt

"""
Usage:

python precision_recall <csv.ext>
eg. python precision_recall "results.csv"

Creates a precision recall curve. The plot is saved in the local directory.
"""

def main():
    
    argc = len(sys.argv)
    
    if argc < 2:
        print "specifiy an input file"
        return

    results = sys.argv[1]
		
	# load data
	data = np.loadtxt(open(results), delimiter=',', skiprows=1)
	precision = data[:,4]
	recall = data[:,5]
	
	# plot
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.xlabel("Recall")
	plt.xlim([0,1.1])
	plt.ylabel("Precision")
	plt.ylim([0,1.1])
	plt.title("Precision Recall") 
	plt.grid(True)
	plt.plot(recall,precision,'r')
	plt.savefig("PrecisionRecall.png")
		    	
    return

if __name__ == "__main__":
    main()
