from distutils.core import setup

with open('.version', 'r') as f:
    version = f.readline()

setup(name='Validation',
      version=version,
      description='Packege for validators',
      author='DreamTeam',
      author_email='IAPodpruzhnikov@sberbank.ru',
      url='https://stash.sigma.sbrf.ru/',
      packages=['distutils', 'distutils.command'],
     )