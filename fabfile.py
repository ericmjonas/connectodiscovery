from fabric.api import local, env, run, put, cd, task
from fabric.contrib import project


from sparkutil import sparkfab as s

s.config['cluster_name'] = 'netmotifs' # ALWAYS SET THIS
env.roledefs = s.create_roles()

@task
def deploy_files():
    put('netmotifs.egg', '/data/netmotifs.egg')

@task
def add_packages():
    env['forward_agent']= True
    run("conda update --all --yes")
    run("pip install ruffus")
    run("pip install --upgrade boto")
    run("pip install colorbrewer")
    run("pip install multyvac")
    run("pip install fabric")
    run("pip install --upgrade git+ssh://git@github.com/ericmjonas/sparkutils.git")
    run("yum -y install tmux")
    run("yum -y update")

@task
def deploy():
    """
    Uses rsync. Note directory trailing-slash behavior is anti-intuitive
    """
    local('git ls-tree --full-tree --name-only -r HEAD > .git-files-list')
    
    project.rsync_project("/data/connect-disco-paper/", local_dir="./",
                          exclude=['*.pickle', 'experiments/synthdifferent/sparkdata'],
                          extra_opts='--files-from=.git-files-list')
    
    env['forward_agent']= True
    with cd("/data/connect-disco-paper"):
        run('~/spark-ec2/copy-dir .')

    project.rsync_project("/data/preprocess/", local_dir="../preprocess/")

""""
What's the deal with the multivac error
org.apache.spark.api.python.PythonException: Traceback (most recent call last):
  File "/root/spark/python/pyspark/worker.py", line 75, in main
    command = pickleSer._read_with_length(infile)
  File "/root/spark/python/pyspark/serializers.py", line 150, in _read_with_length
    return self.loads(obj)
  File "/root/spark/python/pyspark/cloudpickle.py", line 811, in subimport
    __import__(name)
  File "./netmotifs.egg/irm/__init__.py", line 10, in <module>
    import experiments
  File "./netmotifs.egg/irm/experiments.py", line 11, in <module>
  File "/opt/anaconda/lib/python2.7/site-packages/multyvac/__init__.py", line 27, in <module>
    _multyvac = Multyvac()
  File "/opt/anaconda/lib/python2.7/site-packages/multyvac/multyvac.py", line 81, in __init__
    self.config = ConfigModule(self, api_key, api_secret_key, api_url)
  File "/opt/anaconda/lib/python2.7/site-packages/multyvac/config.py", line 33, in __init__
    self._load_config()
  File "/opt/anaconda/lib/python2.7/site-packages/multyvac/config.py", line 146, in _load_config
    conf = json.load(f)
  File "/opt/anaconda/lib/python2.7/json/__init__.py", line 290, in load
    **kw)
  File "/opt/anaconda/lib/python2.7/json/__init__.py", line 338, in loads
    return _default_decoder.decode(s)
  File "/opt/anaconda/lib/python2.7/json/decoder.py", line 366, in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
  File "/opt/anaconda/lib/python2.7/json/decoder.py", line 384, in raw_decode
    raise ValueError("No JSON object could be decoded")
ValueError: (ValueError('No JSON object could be decoded',), <function subimport at 0x7fb1e007e6e0>, ('irm',))

        org.apache.spark.api.python.PythonRDD$$anon$1.read(PythonRDD.scala:124)
        org.apache.spark.api.python.PythonRDD$$anon$1.<init>(PythonRDD.scala:154)
        org.apache.spark.api.python.PythonRDD.compute(PythonRDD.scala:87)
        org.apache.spark.rdd.RDD.computeOrReadCheckpoint(RDD.scala:262)
        org.apache.spark.rdd.RDD.iterator(RDD.scala:229)
        org.apache.spark.rdd.MapPartitionsRDD.compute(MapPartitionsRDD.scala:35)
        org.apache.spark.rdd.RDD.computeOrReadCheckpoint(RDD.scala:262)
        org.apache.spark.rdd.RDD.iterator(RDD.scala:229)
        org.apache.spark.rdd.MappedRDD.compute(MappedRDD.scala:31)
        org.apache.spark.rdd.RDD.computeOrReadCheckpoint(RDD.scala:262)
        org.apache.spark.rdd.RDD.iterator(RDD.scala:229)
        org.apache.spark.scheduler.ResultTask.runTask(ResultTask.scala:62)
        org.apache.spark.scheduler.Task.run(Task.scala:54)
        org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:177)
        java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1145)
        java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:615)
        java.lang.Thread.run(Thread.java:745)


"""


@task
def get_images():
    project.rsync_project("/data/connect-disco-paper/experiments", local_dir="./",
                          extra_opts="--include '*.pdf' --include='*/' --exclude='*' ",
                          upload=False)
    project.rsync_project("/data/connect-disco-paper/experiments", local_dir="./",
                          extra_opts="--include '*.pickle' --include='*/' --exclude='*' ",
                          upload=False)
    

@task
def get_data():
    """
    Copy the data from remote to local
    """

    DATASETS = [('experiments/mouseretina/sparkdatacv',
                 'experiments/mouseretina/',
                 ['predlinks.pickle', 'assign.pickle']),
                
                 ('experiments/synthdifferent/sparkdata',
                  'experiments/synthdifferent/',
                  ['predlinks.pickle', 'assign.pickle'])]
    
   
    for remote_dir, local_dir, globs in DATASETS:
        
        for g in globs:
            project.rsync_project("/data/connect-disco-paper/" + remote_dir, 
                                  local_dir=local_dir, 
                                  extra_opts="--include '%s' --include='*/' --exclude='*' " % g,
                              
                                  upload=False)
    
