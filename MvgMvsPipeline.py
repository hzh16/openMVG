#!/usr/bin/python3
# -*- encoding: utf-8 -*-

import os
import subprocess
import sys
import argparse

DEBUG = False

if sys.platform.startswith('win'):
    PATH_DELIM = ';'
    FOLDER_DELIM = '\\'
else:
    PATH_DELIM = ':'
    FOLDER_DELIM = '/'

# add this script's directory to PATH
os.environ['PATH'] += PATH_DELIM + os.path.dirname(os.path.abspath(__file__))

# add current directory to PATH
os.environ['PATH'] += PATH_DELIM + os.getcwd()

# change to your own path
#OPENMVG_BIN = "/home/zihang/denseR/openMVG/openMVG_Build/Linux-x86_64-RELEASE"
#OPENMVG_BIN = "/home/zihang/stevens-master/pose_estimation/openMVG/build/Linux-x86_64-RELEASE"
OPENMVG_BIN1 = "/home/zihang/stevens-master/pose_estimation/openMVG/build/Linux-x86_64-RELEASE"
OPENMVG_BIN = "/home/zihang/openMVG_deep/openMVG_Build/Linux-x86_64-RELEASE"
#OPENMVG_BIN = "/home/zihang/openMVG_deep/openMVG_build2/Linux-x86_64-RELEASE"
OPENMVG_PYTHONSRC = "/home/zihang/openMVG_deep/openMVG/src/software/SfM/python/external_features_demo"
#descriptor_type = "DeepLabv3"
descriptor_type = "DINOv2"
#OPENMVS_BIN = "/home/zihang/denseR/openMVS/openMVS_build/bin"
OPENMVS_BIN = "/home/zihang/dense/openMVS/bin/"
CAMERA_SENSOR_DB_DIRECTORY = "/home/zihang/denseR/openMVG/src/openMVG/exif/sensor_width_database"


CAMERA_SENSOR_DB_FILE = "sensor_width_camera_database.txt"

PRESET = {'SEQUENTIAL': [0, 1, 2, 3, 4, 5, 7, 11, 12, 13, 14, 15],
          'SEQUENTIAL_2': [0, 1, 2, 3, 4, 5],
          'SEQUENTIAL_PAIRWISE': [0, 1, 2, 3, 25],
          'SEQUENTIAL_21': [0, 1],
          'SEQUENTIAL_22': [2, 3, 4, 5],
          'SEQUENTIAL_DEEP_21': [0, 18],
          'SEQUENTIAL_DEEP_22': [2, 19, 20, 5],
          'SEQUENTIAL_DISK_DEEP_PAIRWISE': [0, 18, 2, 19, 26],
          'SEQUENTIAL_DEEP_PAIRWISE': [0, 1, 27, 2, 19, 20, 5],
          'SEQUENTIAL_SIFT_DEEP': [0, 1, 27, 2, 3, 4, 5],
          'GLOBAL': [0, 1, 2, 3, 4, 6, 11, 12, 13, 14, 15],
          'MVG_SEQ': [0, 1, 2, 3, 4, 5, 7, 8, 9, 11],
          'MVG_GLOBAL': [0, 1, 2, 3, 4, 6, 7, 8, 9, 11],
          'MVS': [12, 13, 14, 15],
          'MVS_SGM': [16, 17],
          'SEQUENTIAL_DEEP': [0, 18, 2, 19, 20, 5, 7, 11, 12, 13, 14, 15],
          'SEQUENTIAL_DISK': [0, 21, 22, 4, 5, 7, 11, 12, 13, 14, 15],
          'SEQUENTIAL_BOTH': [0, 1, 23, 24, 4, 5, 7, 11, 12, 13, 14, 15]
          }

PRESET_DEFAULT = 'SEQUENTIAL_2'

# HELPERS for terminal colors
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
NO_EFFECT, BOLD, UNDERLINE, BLINK, INVERSE, HIDDEN = (0, 1, 4, 5, 7, 8)


# from Python cookbook, #475186
def has_colours(stream):
    '''
        Return stream colours capability
    '''
    if not hasattr(stream, "isatty"):
        return False
    if not stream.isatty():
        return False  # auto color only on TTYs
    try:
        import curses
        curses.setupterm()
        return curses.tigetnum("colors") > 2
    except Exception:
        # guess false in case of error
        return False

HAS_COLOURS = has_colours(sys.stdout)


def printout(text, colour=WHITE, background=BLACK, effect=NO_EFFECT):
    """
        print() with colour
    """
    if HAS_COLOURS:
        seq = "\x1b[%d;%d;%dm" % (effect, 30+colour, 40+background) + text + "\x1b[0m"
        sys.stdout.write(seq+'\r\n')
    else:
        sys.stdout.write(text+'\r\n')


# OBJECTS to store config and data in
class ConfContainer:
    """
        Container for all the config variables
    """
    def __init__(self):
        pass


class AStep:
    """ Represents a process step to be run """
    def __init__(self, info, cmd, opt):
        self.info = info
        self.cmd = cmd
        self.opt = opt


class StepsStore:
    """ List of steps with facilities to configure them """
    def __init__(self):
        self.steps_data = [
            ["Intrinsics analysis",          # 0
             os.path.join(OPENMVG_BIN1, "openMVG_main_SfMInit_ImageListing"),
             ["-i", "%input_dir%", "-o", "%matches_dir%", "-d", "%camera_file_params%"]],
            ["Compute features",             # 1
             os.path.join(OPENMVG_BIN, "openMVG_main_ComputeFeatures"),
             ["-i", "%matches_dir%"+FOLDER_DELIM+"sfm_data.json", "-o", "%matches_dir%", "-m", "SIFT", "-p", "HIGH"]],
            ["Compute pairs",                # 2
             os.path.join(OPENMVG_BIN, "openMVG_main_PairGenerator"),
             ["-i", "%matches_dir%"+FOLDER_DELIM+"sfm_data.json", "-o", "%matches_dir%"+FOLDER_DELIM+"pairs.bin"]],
            ["Compute matches",              # 3
             os.path.join(OPENMVG_BIN, "openMVG_main_ComputeMatches"),
             ["-i", "%matches_dir%"+FOLDER_DELIM+"sfm_data.json", "-p", "%matches_dir%"+FOLDER_DELIM+"pairs.bin", "-o", "%matches_dir%"+FOLDER_DELIM+"matches.putative.bin"]],
            ["Filter matches",               # 4
             os.path.join(OPENMVG_BIN, "openMVG_main_GeometricFilter"),
             ["-i", "%matches_dir%"+FOLDER_DELIM+"sfm_data.json", "-m", "%matches_dir%"+FOLDER_DELIM+"matches.putative.bin", "-o", "%matches_dir%"+FOLDER_DELIM+"matches.f.bin"]],
            ["Incremental reconstruction",   # 5
             os.path.join(OPENMVG_BIN, "openMVG_main_SfM"),
             ["-i", "%matches_dir%"+FOLDER_DELIM+"sfm_data.json", "-m", "%matches_dir%", "-o", "%reconstruction_dir%", "-s", "INCREMENTAL"]],
            ["Global reconstruction",        # 6
             os.path.join(OPENMVG_BIN, "openMVG_main_SfM"),
             ["-i", "%matches_dir%"+FOLDER_DELIM+"sfm_data.json", "-m", "%matches_dir%", "-o", "%reconstruction_dir%", "-s", "GLOBAL", "-M", "%matches_dir%"+FOLDER_DELIM+"matches.e.bin"]],
            ["Colorize Structure",           # 7
             os.path.join(OPENMVG_BIN, "openMVG_main_ComputeSfM_DataColor"),
             ["-i", "%reconstruction_dir%"+FOLDER_DELIM+"sfm_data.bin", "-o", "%reconstruction_dir%"+FOLDER_DELIM+"colorized.ply"]],
            ["Structure from Known Poses",   # 8
             os.path.join(OPENMVG_BIN, "openMVG_main_ComputeStructureFromKnownPoses"),
             ["-i", "%reconstruction_dir%"+FOLDER_DELIM+"sfm_data.bin", "-m", "%matches_dir%", "-f", "%matches_dir%"+FOLDER_DELIM+"matches.f.bin", "-o", "%reconstruction_dir%"+FOLDER_DELIM+"robust.bin"]],
            ["Colorized robust triangulation",  # 9
             os.path.join(OPENMVG_BIN, "openMVG_main_ComputeSfM_DataColor"),
             ["-i", "%reconstruction_dir%"+FOLDER_DELIM+"robust.bin", "-o", "%reconstruction_dir%"+FOLDER_DELIM+"robust_colorized.ply"]],
            ["Control Points Registration",  # 10
             os.path.join(OPENMVG_BIN, "ui_openMVG_control_points_registration"),
             ["-i", "%reconstruction_dir%"+FOLDER_DELIM+"sfm_data.bin"]],
            ["Export to openMVS",            # 11
             os.path.join(OPENMVG_BIN, "openMVG_main_openMVG2openMVS"),
             ["-i", "%reconstruction_dir%"+FOLDER_DELIM+"sfm_data.bin", "-o", "%mvs_dir%"+FOLDER_DELIM+"scene.mvs", "-d", "%mvs_dir%"+FOLDER_DELIM+"images"]],
            ["Densify point cloud",          # 12
             os.path.join(OPENMVS_BIN, "DensifyPointCloud"),
             ["scene.mvs", "--dense-config-file", "Densify.ini", "--resolution-level", "1", "--number-views", "8", "-w", "\"%mvs_dir%\""]],
            ["Reconstruct the mesh",         # 13
             os.path.join(OPENMVS_BIN, "ReconstructMesh"),
             ["scene_dense.mvs", "-p", "scene_dense.ply", "-w", "\"%mvs_dir%\""]],
            ["Refine the mesh",              # 14
             os.path.join(OPENMVS_BIN, "RefineMesh"),
             ["scene_dense.mvs", "-m", "scene_dense_mesh.ply", "-o", "scene_dense_mesh_refine.mvs", "--scales", "1", "--gradient-step", "25.05", "-w", "\"%mvs_dir%\""]],
            ["Texture the mesh",             # 15
             os.path.join(OPENMVS_BIN, "TextureMesh"),
             ["scene_dense.mvs", "-m", "scene_dense_mesh_refine.ply", "-o", "scene_dense_mesh_refine_texture.mvs", "--decimate", "0.5", "-w", "\"%mvs_dir%\""]],
            ["Estimate disparity-maps",      # 16
             os.path.join(OPENMVS_BIN, "DensifyPointCloud"),
             ["scene.mvs", "--dense-config-file", "Densify.ini", "--fusion-mode", "-1", "-w", "\"%mvs_dir%\""]],
            ["Fuse disparity-maps",          # 17
             os.path.join(OPENMVS_BIN, "DensifyPointCloud"),
             ["scene.mvs", "--dense-config-file", "Densify.ini", "--fusion-mode", "-2", "-w", "\"%mvs_dir%\""]],
            ["Compute deep features",        # 18
             "python", 
             [os.path.join(OPENMVG_PYTHONSRC, "kornia_deep_features.py"),
             "-i", "%matches_dir%"+FOLDER_DELIM+"sfm_data.json", "-m", "%matches_dir%", "--deep_descriptor_type", descriptor_type, "--max_resolution", "512"]],
            ["Compute matches deep feature", # 19
             os.path.join(OPENMVG_BIN, "openMVG_main_ComputeMatches"),
             ["-i", "%matches_dir%"+FOLDER_DELIM+"sfm_data.json", "-p", "%matches_dir%"+FOLDER_DELIM+"pairs.bin", "-o", "%matches_dir%"+FOLDER_DELIM+"matches.putative.bin", "-d", descriptor_type]],
            ["Filter matches deep feature",  # 20
             os.path.join(OPENMVG_BIN, "openMVG_main_GeometricFilter"),
             ["-i", "%matches_dir%"+FOLDER_DELIM+"sfm_data.json", "-m", "%matches_dir%"+FOLDER_DELIM+"matches.putative.bin", "-o", "%matches_dir%"+FOLDER_DELIM+"matches.f.bin", "-d", descriptor_type]],
            ["Compute disk features",        # 21
             "python", 
             [os.path.join(OPENMVG_PYTHONSRC, "kornia_demo.py"),
             "-i", "%matches_dir%"+FOLDER_DELIM+"sfm_data.json", "-m", "%matches_dir%", "--preset", "EXTRACT"]],
            ["Compute match disk feature",   # 22
             "python", 
             [os.path.join(OPENMVG_PYTHONSRC, "kornia_demo.py"),
             "-i", "%matches_dir%"+FOLDER_DELIM+"sfm_data.json", "-m", "%matches_dir%", "--preset", "MATCH"]],
            ["Compute two features",        # 23
             "python",
             [os.path.join(OPENMVG_PYTHONSRC, "kornia_two_features.py"),
             "-i", "%matches_dir%"+FOLDER_DELIM+"sfm_data.json", "-m", "%matches_dir%", "--preset", "EXTRACT"]],
            ["Compute match disk feature",   # 24
             "python",
             [os.path.join(OPENMVG_PYTHONSRC, "kornia_two_features.py"),
             "-i", "%matches_dir%"+FOLDER_DELIM+"sfm_data.json", "-m", "%matches_dir%", "--preset", "MATCH"]],
            ["Filter matches with poses",    # 25
             os.path.join(OPENMVG_BIN, "openMVG_main_GeometricFilter"),
             ["-i", "%matches_dir%" + FOLDER_DELIM + "sfm_data.json", "-m", "%matches_dir%" + FOLDER_DELIM + "matches.putative.bin", "-o",
              "%matches_dir%" + FOLDER_DELIM + "matches.f.bin", "-g", "e"]],
            ["Filter matches with deep and pose", # 26
             os.path.join(OPENMVG_BIN, "openMVG_main_GeometricFilter"),
             ["-i", "%matches_dir%" + FOLDER_DELIM + "sfm_data.json", "-m", "%matches_dir%" + FOLDER_DELIM + "matches.putative.bin", "-o",
              "%matches_dir%" + FOLDER_DELIM + "matches.f.bin", "-g", "e", "-d", descriptor_type]],
            ["Compute deep feature on sift",   # 27
             "python",
             [os.path.join(OPENMVG_PYTHONSRC, "kornia_two_features.py"),
             "-i", "%matches_dir%"+FOLDER_DELIM+"sfm_data.json", "-m", "%matches_dir%", "--preset", "EXTRACT_REPLACE", "--deep_descriptor_type", descriptor_type]],

            ]


    def __getitem__(self, indice):
        return AStep(*self.steps_data[indice])

    def length(self):
        return len(self.steps_data)

    def apply_conf(self, conf):
        """ replace each %var% per conf.var value in steps data """
        for s in self.steps_data:
            o2 = []
            for o in s[2]:
                co = o.replace("%input_dir%", conf.input_dir)
                co = co.replace("%output_dir%", conf.output_dir)
                co = co.replace("%matches_dir%", conf.matches_dir)
                co = co.replace("%reconstruction_dir%", conf.reconstruction_dir)
                co = co.replace("%mvs_dir%", conf.mvs_dir)
                co = co.replace("%camera_file_params%", conf.camera_file_params)
                o2.append(co)
            s[2] = o2

    def replace_opt(self, idx, str_exist, str_new):
        """ replace each existing str_exist with str_new per opt value in step idx data """
        s = self.steps_data[idx]
        o2 = []
        for o in s[2]:
            co = o.replace(str_exist, str_new)
            o2.append(co)
        s[2] = o2


CONF = ConfContainer()
STEPS = StepsStore()

# ARGS
PARSER = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Photogrammetry reconstruction with these steps: \r\n" +
    "\r\n".join(("\t%i. %s\t %s" % (t, STEPS[t].info, STEPS[t].cmd) for t in range(STEPS.length())))
    )
PARSER.add_argument('input_dir',
                    help="the directory which contains the pictures set.")
PARSER.add_argument('output_dir',
                    help="the directory which will contain the resulting files.")
PARSER.add_argument('--steps',
                    type=int,
                    nargs="+",
                    help="steps to process")
PARSER.add_argument('--preset',
                    help="steps list preset in \r\n" +
                    " \r\n".join([k + " = " + str(PRESET[k]) for k in PRESET]) +
                    " \r\ndefault : " + PRESET_DEFAULT)
PARSER.add_argument('--openmvg_bin',
                    type=str,
                    help="openmvg bin directory",
                    default=None)

GROUP = PARSER.add_argument_group('Passthrough', description="Option to be passed to command lines (remove - in front of option names)\r\ne.g. --1 p ULTRA to use the ULTRA preset in openMVG_main_ComputeFeatures\r\nFor example, running the script as follows,\r\nMvgMvsPipeline.py input_dir output_dir --1 p HIGH n 8 --3 n ANNL2\r\nwhere --1 refer to openMVG_main_ComputeFeatures, p refers to\r\ndescriberPreset option which HIGH was chosen, and n refers to\r\nnumThreads which 8 was used. --3 refer to second step (openMVG_main_ComputeMatches),\r\nn refers to nearest_matching_method option which ANNL2 was chosen")
for n in range(STEPS.length()):
    GROUP.add_argument('--'+str(n), nargs='+')

PARSER.parse_args(namespace=CONF)  # store args in the ConfContainer


# FOLDERS

def mkdir_ine(dirname):
    """Create the folder if not presents"""
    if not os.path.exists(dirname):
        os.mkdir(dirname)


# Absolute path for input and output dirs
CONF.input_dir = os.path.abspath(CONF.input_dir)
CONF.output_dir = os.path.abspath(CONF.output_dir)

if not os.path.exists(CONF.input_dir):
    sys.exit("%s: path not found" % CONF.input_dir)

if type(CONF.openmvg_bin) != type(None):
    OPENMVG_BIN=CONF.openmvg_bin

CONF.reconstruction_dir = os.path.join(CONF.output_dir, "sfm")
CONF.matches_dir = os.path.join(CONF.reconstruction_dir, "matches")
CONF.mvs_dir = os.path.join(CONF.output_dir, "mvs")
CONF.camera_file_params = os.path.join(CAMERA_SENSOR_DB_DIRECTORY, CAMERA_SENSOR_DB_FILE)

mkdir_ine(CONF.output_dir)
mkdir_ine(CONF.reconstruction_dir)
mkdir_ine(CONF.matches_dir)
mkdir_ine(CONF.mvs_dir)

# Update directories in steps commandlines
STEPS.apply_conf(CONF)

# PRESET
if CONF.steps and CONF.preset:
    sys.exit("Steps and preset arguments can't be set together.")
elif CONF.preset:
    try:
        CONF.steps = PRESET[CONF.preset]
    except KeyError:
        sys.exit("Unknown preset %s, choose %s" % (CONF.preset, ' or '.join([s for s in PRESET])))
elif not CONF.steps:
    CONF.steps = PRESET[PRESET_DEFAULT]

# WALK
print("# Using input dir:  %s" % CONF.input_dir)
print("#      output dir:  %s" % CONF.output_dir)
print("# Steps:  %s" % str(CONF.steps))

if 4 in CONF.steps:    # GeometricFilter
    if 6 in CONF.steps:  # GlobalReconstruction
        # Set the geometric_model of ComputeMatches to Essential
        STEPS.replace_opt(4, FOLDER_DELIM+"matches.f.bin", FOLDER_DELIM+"matches.e.bin")
        STEPS[4].opt.extend(["-g", "e"])

if 20 in CONF.steps:    # TextureMesh
    if 19 not in CONF.steps:  # RefineMesh
        # RefineMesh step is not run, use ReconstructMesh output
        STEPS.replace_opt(20, "scene_dense_mesh_refine.ply", "scene_dense_mesh.ply")
        STEPS.replace_opt(20, "scene_dense_mesh_refine_texture.mvs", "scene_dense_mesh_texture.mvs")

for cstep in CONF.steps:
    printout("#%i. %s" % (cstep, STEPS[cstep].info), effect=INVERSE)

    # Retrieve "passthrough" commandline options
    opt = getattr(CONF, str(cstep))
    if opt:
        # add - sign to short options and -- to long ones
        for o in range(0, len(opt), 2):
            if len(opt[o]) > 1:
                opt[o] = '-' + opt[o]
            opt[o] = '-' + opt[o]
    else:
        opt = []

    # Remove STEPS[cstep].opt options now defined in opt
    for anOpt in STEPS[cstep].opt:
        if anOpt in opt:
            idx = STEPS[cstep].opt.index(anOpt)
            if DEBUG:
                print('#\tRemove ' + str(anOpt) + ' from defaults options at id ' + str(idx))
            del STEPS[cstep].opt[idx:idx+2]

    # create a commandline for the current step
    cmdline = [STEPS[cstep].cmd] + STEPS[cstep].opt + opt
    print('Cmd: ' + ' '.join(cmdline))

    if not DEBUG:
        # Launch the current step
        try:
            pStep = subprocess.Popen(cmdline)
            pStep.wait()
            if pStep.returncode != 0:
                break
        except KeyboardInterrupt:
            sys.exit('\r\nProcess canceled by user, all files remains')
    else:
        print('\t'.join(cmdline))

printout("# Pipeline end #", effect=INVERSE)
