from pyspark import SparkContext

sc = SparkContext()
tmpFile = "sparkdata/srm.data.samples"

r = sc.pickleFile(tmpFile, 5).collect()

