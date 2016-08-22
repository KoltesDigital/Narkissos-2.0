import os

vars = Variables('config.py')

base = Environment(ENV = os.environ, variables = vars, tools = ['mingw'])
base.Append(CPPPATH = ['C:\\opencv\\modules\\core\\include', 'C:\\opencv\\modules\\objdetect\\include', 'C:\\opencv\\modules\\highgui\\include', 'C:\\opencv\\modules\\imgproc\\include', 'C:\\DevIL\\include'])
base.Append(LIBPATH = ['C:\\opencv\\build\\x86\\mingw\\lib', 'C:\\DevIL\\lib'])
base.Append(LIBS = ['opencv_core242', 'opencv_objdetect242', 'opencv_highgui242', 'opencv_imgproc242', 'pthread', 'glfw', 'opengl32', 'DevIL', 'ILU', 'ILUT', 'gdi32'])
base.Append(CPPFLAGS = ['-Wall', '-std=c++0x'])
base.Append(LINKFLAGS = [])

Export('base')

Help("""
Type 'scons [target]' where target may be
  release (default)
  all
""")

release = SConscript('SConscript.release', variant_dir='build/release', duplicate=0)

Alias('all', '.')
Alias('release', release)
Default(release)
