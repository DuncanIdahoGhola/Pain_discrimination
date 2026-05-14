#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on May 13, 2026, at 15:03
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

import psychopy
psychopy.useVersion('2023.2.3')


# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# Run 'Before Experiment' code from compile
import pandas as pd
# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = '2024_RDpaindiscrimination_calibration'  # from the Builder filename that created this script
expInfo = {
    'participant': 'sub-000',
    'com_thermode': 'COM3',
    'nramps': '3',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s/Calibration/%s_%s_%s' % (expInfo['participant'], data.getDateStr(format="%Y-%m-%d"), expInfo['participant'], expName, expInfo['date'] )
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\test\\Desktop\\RD_2024\\01_2024_RDpaindiscrimination_calibration_v5_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.WARNING)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.WARNING)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1920, 1080], fullscr=True, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color='[0, 0, 0]', colorSpace='named',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = '[0, 0, 0]'
        win.colorSpace = 'named'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = True
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='ptb')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='PsychToolbox')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Welcome" ---
    textwelcome = visual.TextStim(win=win, name='textwelcome',
        text='Bienvenue dans le programme de calibration.\n\nVeuillez vous assurer que le stimulateur est bien branché et ouvert.\n\n\nUne fois les vérifications faites, appuyez sur "p".',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard()
    # Run 'Begin Experiment' code from init_thermode
    # Import packages for thermode
    import serial
    import time
    from pytcsii import tcsii_serial
    import threading
    
    
    # Initialize thermode
    port_thermode = tcsii_serial(str(expInfo['com_thermode']))
    port_thermode.set_baseline(38)
    baseline = 38
    rise_time = 0.75
    port_thermode.port.write('Ue00000'.encode())
    
    #Set White Circle
    polygon_1 = visual.Circle(
        win=win, name='polygon',
        size=(0.5, 0.5), 
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor= 'white', fillColor= 'white',
        opacity=1.0, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "instructions2" ---
    text_5 = visual.TextStim(win=win, name='text_5',
        text="Dans la première phase, vous recevrez une série de stimulations augmentant en intensité. Nous débuterons avec des stimulations très faibles (probablement non douloureuses) et l’intensité augmentera graduellement. Après chaque stimulation, vous devrez évaluer verbalement l’intensité de la douleur ressentie. Si vous n’avez pas ressenti de douleur, veuillez déplacer le curseur à l'extrémité gauche de l'échelle.\n\nAprès chaque stimulation thermique, nous vous demanderons également si vous pensez être capable de tolérer la prochaine stimulation. Si vous pensez ne pas être capable de tolérer la prochaine stimulation, nous arrêterons la série. Si vous pensez être en mesure de la tolérer, nous continuerons avec une stimulation d’une intensité plus forte. Notez que vous allez recevoir plusieurs stimulations pendant l’expérience, donc veillez à arrêter à une intensité que vous serez en mesure de tolérer plusieurs fois.\n\nNous allons répéter ce processus 3 fois.\n\nAprès cette phase, l’intensité maximale que vous aurez tolérée sera utilisée comme point de repère pour différentes intensités pour l’expérience au complet et vous ne recevrez jamais une stimulation excédant cette intensité.\n\nDans la seconde phase, vous allez recevoir des stimulations d’intensités aléatoires sous votre niveau de tolérance et vous devrez évaluer la douleur ressentie après chaque choc.\n\nAppuyer sur la touche spéciale pour débuter",
        font='Open Sans',
        pos=(0, 0), height=0.032, wrapWidth=1.3, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_5 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "instructions_start_ramp" ---
    text = visual.TextStim(win=win, name='text',
        text='Prêt à commencer une série de stimulations.\n\nExpérimentateur :\nAssurez-vous que le stimulateur est connecté.\n\nAppuyer sur la touche spéciale pour commencer.',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_2 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "confirm_continue" ---
    # Run 'Begin Experiment' code from set_therm_2
    
    # Initialize all the counter to 0
    curr_item = -1
    curr_item_loc = -1
    
    # Choose the font size for thermode location indicator
    loctherm_size = 0.02
    
    
    # Ramp intensities
    therm_intensity = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    
    # Inttialize empty lists
    intensities_list = []*3
    ratings_list = []*3
    accept_list = []*3
    fired_list = []*3
    text_12 = visual.TextStim(win=win, name='text_12',
        text='Appuyer sur le bouton pour la prochaine stimulation.',
        font='Arial',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    text_10 = visual.TextStim(win=win, name='text_10',
        text='',
        font='Open Sans',
        pos=(-.70, -.45), height=loctherm_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_8 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "wait" ---
    text_3 = visual.TextStim(win=win, name='text_3',
        text='',
        font='Arial',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    text_14 = visual.TextStim(win=win, name='text_14',
        text='',
        font='Open Sans',
        pos=(-.70, -.45), height=loctherm_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "therm_trig" ---
    text_15 = visual.TextStim(win=win, name='text_15',
        text='',
        font='Open Sans',
        pos=(-.70, -.45), height=loctherm_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "rating_scale" ---
    # Run 'Begin Experiment' code from slidercode
    # Initialize keyboard
    kb = keyboard.Keyboard()
    
    ratingScale = visual.RatingScale(win=win, name='ratingScale', lineColor=(255, 255, 255), low=0, high=1000, precision=1000, size=1, tickMarks=None, tickHeight=1, scale=None, labels=None, marker=None, markerColor=(255, 255, 255), markerStart=0.5, textColor=(255, 255, 255), pos=(0, 0), stretch=2, showValue=None, showAccept=None, textSize=1.2)
    main_text = visual.TextStim(win=win, name='main_text',
        text="\nVeuillez évaluer l'intensité de la stimulation que vous venez de recevoir.",
        font='Arial',
        pos=(0, 0.4), height=0.07, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-2.0);
    confirm_text = visual.TextStim(win=win, name='confirm_text',
        text='Appuyer sur le bouton du Haut pour poursuivre.',
        font='Arial',
        pos=(0, -0.2), height=0.05, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-3.0);
    text_16 = visual.TextStim(win=win, name='text_16',
        text='',
        font='Open Sans',
        pos=(-.70, -.45), height=loctherm_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "press_when_ready" ---
    # Run 'Begin Experiment' code from log_code
    series_trial = []
    ramp_trial = []
    intensities_list = []
    ratings_list = []
    fired_list = []
    text_2 = visual.TextStim(win=win, name='text_2',
        text='Si vous pensez être capable de tolérer la prochaine stimulation, dites à l’expérimentatrice de continuer.\n\n\nSi vous ne pensez pas être capable de tolérer la prochaine stimulation, dites à l’expérimentatrice d’arrêter ici. \n\n\n\n',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_3 = keyboard.Keyboard()
    text_17 = visual.TextStim(win=win, name='text_17',
        text='',
        font='Open Sans',
        pos=(-.70, -.45), height=loctherm_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "bloc_terminé" ---
    block_done = visual.TextStim(win=win, name='block_done',
        text='Bloc terminé',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "compile_ramps" ---
    premiere_partie = visual.TextStim(win=win, name='premiere_partie',
        text='Première partie de la calibration terminée\n\n',
        font='Arial',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "deuxieme_start" ---
    debut_deuxieme = visual.TextStim(win=win, name='debut_deuxieme',
        text='Deuxième phase. \n\nDans la seconde phase, vous allez recevoir des stimulations d’intensité aléatoire sous votre niveau de tolérance et vous devrez évaluer la douleur ressentie après chaque stimulation.\n\nAppuyer sur la touche spéciale pour débuter.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_10 = keyboard.Keyboard()
    # Run 'Begin Experiment' code from load_csv
    
    
    #intensity_list = [48.0, 45.3, 44.9, 43.4, 47.6, 43.0, 45.7, 44.2, 43.8, 46.1, 46.5, 44.5, 47.2, 46.8]
    
    # --- Initialize components for Routine "press_when_ready2" ---
    text_9 = visual.TextStim(win=win, name='text_9',
        text='Appuyer sur le bouton pour la prochaine stimulation.',
        font='Arial',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_7 = keyboard.Keyboard()
    # Run 'Begin Experiment' code from code_part_2
    curr_item_scd = -1
    text_18 = visual.TextStim(win=win, name='text_18',
        text='',
        font='Open Sans',
        pos=(-.70, -.45), height=loctherm_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "wait_2" ---
    wait_text2 = visual.TextStim(win=win, name='wait_text2',
        text='',
        font='Arial',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    text_19 = visual.TextStim(win=win, name='text_19',
        text='',
        font='Open Sans',
        pos=(-.70, -.45), height=loctherm_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "therm_trig2" ---
    text_20 = visual.TextStim(win=win, name='text_20',
        text='',
        font='Open Sans',
        pos=(-.70, -.45), height=loctherm_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "rating_scale2" ---
    keyResp_2 = keyboard.Keyboard()
    # Run 'Begin Experiment' code from slidercode_2
    kb = keyboard.Keyboard()
    win.mouseVisible = False
    intensities_random = []
    ratings_random =[]
    ratingScale_2 = visual.RatingScale(win=win, name='ratingScale_2', lineColor=(255, 255, 255), low=0, high=1000, precision=1000, size=1, tickMarks=None, tickHeight=1, scale=None, labels=None, marker=None, markerColor=(255, 255, 255), markerStart=0.5, textColor=(255, 255, 255), pos=(0, 0), stretch=2, showValue=None, showAccept=None, textSize=1.2)
    main_text_2 = visual.TextStim(win=win, name='main_text_2',
        text="\nVeuillez évaluer l'intensité de la stimulation que vous venez de recevoir.",
        font='Arial',
        pos=(0, 0.4), height=0.07, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-3.0);
    confirm_text_2 = visual.TextStim(win=win, name='confirm_text_2',
        text='Appuyer sur ESPACE pour poursuivre.',
        font='Arial',
        pos=(0, -0.2), height=0.05, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-4.0);
    text_21 = visual.TextStim(win=win, name='text_21',
        text='',
        font='Open Sans',
        pos=(-.70, -.45), height=loctherm_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    
    # --- Initialize components for Routine "thank_you_output_code" ---
    text_13 = visual.TextStim(win=win, name='text_13',
        text='Terminé!\n',
        font='Arial',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from choose_stim_2
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "Welcome" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Welcome.started', globalClock.getTime())
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # Run 'Begin Routine' code from init_thermode
    # Remove mouse
    win.mouseVisible = False
    # keep track of which components have finished
    WelcomeComponents = [textwelcome, key_resp]
    for thisComponent in WelcomeComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Welcome" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *textwelcome* updates
        
        # if textwelcome is starting this frame...
        if textwelcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textwelcome.frameNStart = frameN  # exact frame index
            textwelcome.tStart = t  # local t and not account for scr refresh
            textwelcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textwelcome, 'tStartRefresh')  # time at next scr refresh
            # update status
            textwelcome.status = STARTED
            textwelcome.setAutoDraw(True)
        
        # if textwelcome is active this frame...
        if textwelcome.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in WelcomeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Welcome" ---
    for thisComponent in WelcomeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Welcome.stopped', globalClock.getTime())
    # the Routine "Welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instructions2.started', globalClock.getTime())
    key_resp_5.keys = []
    key_resp_5.rt = []
    _key_resp_5_allKeys = []
    # keep track of which components have finished
    instructions2Components = [text_5, key_resp_5]
    for thisComponent in instructions2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions2" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_5* updates
        
        # if text_5 is starting this frame...
        if text_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_5.frameNStart = frameN  # exact frame index
            text_5.tStart = t  # local t and not account for scr refresh
            text_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_5, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_5.status = STARTED
            text_5.setAutoDraw(True)
        
        # if text_5 is active this frame...
        if text_5.status == STARTED:
            # update params
            pass
        
        # *key_resp_5* updates
        waitOnFlip = False
        
        # if key_resp_5 is starting this frame...
        if key_resp_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_5.frameNStart = frameN  # exact frame index
            key_resp_5.tStart = t  # local t and not account for scr refresh
            key_resp_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_5, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_5.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_5.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_5.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_5.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_5.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_5_allKeys.extend(theseKeys)
            if len(_key_resp_5_allKeys):
                key_resp_5.keys = _key_resp_5_allKeys[-1].name  # just the last key pressed
                key_resp_5.rt = _key_resp_5_allKeys[-1].rt
                key_resp_5.duration = _key_resp_5_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions2" ---
    for thisComponent in instructions2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instructions2.stopped', globalClock.getTime())
    # the Routine "instructions2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    series = data.TrialHandler(nReps=expInfo['nramps'], method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='series')
    thisExp.addLoop(series)  # add the loop to the experiment
    thisSerie = series.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisSerie.rgb)
    if thisSerie != None:
        for paramName in thisSerie:
            globals()[paramName] = thisSerie[paramName]
    
    for thisSerie in series:
        currentLoop = series
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisSerie.rgb)
        if thisSerie != None:
            for paramName in thisSerie:
                globals()[paramName] = thisSerie[paramName]
        
        # --- Prepare to start Routine "instructions_start_ramp" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('instructions_start_ramp.started', globalClock.getTime())
        key_resp_2.keys = []
        key_resp_2.rt = []
        _key_resp_2_allKeys = []
        # Run 'Begin Routine' code from reset_curr
        
        # Start the index counter for thermode location
        curr_item = -1
        
        # Make sure mouse is gone!
        win.mouseVisible = False
        # keep track of which components have finished
        instructions_start_rampComponents = [text, key_resp_2]
        for thisComponent in instructions_start_rampComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "instructions_start_ramp" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # *key_resp_2* updates
            waitOnFlip = False
            
            # if key_resp_2 is starting this frame...
            if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_2.frameNStart = frameN  # exact frame index
                key_resp_2.tStart = t  # local t and not account for scr refresh
                key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_resp_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_2.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_2.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_2_allKeys.extend(theseKeys)
                if len(_key_resp_2_allKeys):
                    key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                    key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                    key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in instructions_start_rampComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "instructions_start_ramp" ---
        for thisComponent in instructions_start_rampComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('instructions_start_ramp.stopped', globalClock.getTime())
        # the Routine "instructions_start_ramp" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        Ramp = data.TrialHandler(nReps=10, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='Ramp')
        thisExp.addLoop(Ramp)  # add the loop to the experiment
        thisRamp = Ramp.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisRamp.rgb)
        if thisRamp != None:
            for paramName in thisRamp:
                globals()[paramName] = thisRamp[paramName]
        
        for thisRamp in Ramp:
            currentLoop = Ramp
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisRamp.rgb)
            if thisRamp != None:
                for paramName in thisRamp:
                    globals()[paramName] = thisRamp[paramName]
            
            # --- Prepare to start Routine "confirm_continue" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('confirm_continue.started', globalClock.getTime())
            # Run 'Begin Routine' code from set_therm_2
            
            # Update counter
            curr_item += 1
            
            # Get current intensity
            intensity = therm_intensity[curr_item]
            
            #Set temp
            curr_temp = intensity
            
            # Calculate rise rate
            rise_curr = np.round((curr_temp - baseline)/rise_time, 1)
            
            # Set thermode
            port_thermode.set_stim(target=curr_temp, rise_rate=rise_curr, return_rate=rise_curr,
                        dur_ms=11500,
                        dur_mode='fixed_total',
                        surfaces=[1, 2, 3, 4, 5])
            fired = 0
            
            # Jitter waiting
            wait1jitter = np.random.choice([2000, 2100, 2200, 2300, 2400, 2500])
            # in secs
            wait1jitter = wait1jitter/1000
            
            # Get phase
            ramp_phase = 1
            
            # Get localisation
            curr_item_loc += 1
            thermode_localisation = 'T: ' + str(curr_item_loc) 
            
            
            
            
                
            
            
            
            
            key_resp_8.keys = []
            key_resp_8.rt = []
            _key_resp_8_allKeys = []
            # keep track of which components have finished
            confirm_continueComponents = [text_12, text_10, key_resp_8]
            for thisComponent in confirm_continueComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "confirm_continue" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_12* updates
                
                # if text_12 is starting this frame...
                if text_12.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_12.frameNStart = frameN  # exact frame index
                    text_12.tStart = t  # local t and not account for scr refresh
                    text_12.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_12, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    text_12.status = STARTED
                    text_12.setAutoDraw(True)
                
                # if text_12 is active this frame...
                if text_12.status == STARTED:
                    # update params
                    pass
                
                # *text_10* updates
                
                # if text_10 is starting this frame...
                if text_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_10.frameNStart = frameN  # exact frame index
                    text_10.tStart = t  # local t and not account for scr refresh
                    text_10.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_10, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    text_10.status = STARTED
                    text_10.setAutoDraw(True)
                
                # if text_10 is active this frame...
                if text_10.status == STARTED:
                    # update params
                    text_10.setText(thermode_localisation
                    , log=False)
                
                # *key_resp_8* updates
                waitOnFlip = False
                
                # if key_resp_8 is starting this frame...
                if key_resp_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_8.frameNStart = frameN  # exact frame index
                    key_resp_8.tStart = t  # local t and not account for scr refresh
                    key_resp_8.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_8, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_8.started')
                    # update status
                    key_resp_8.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_8.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_8.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_8.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_8.getKeys(keyList=['p','n'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_8_allKeys.extend(theseKeys)
                    if len(_key_resp_8_allKeys):
                        key_resp_8.keys = _key_resp_8_allKeys[-1].name  # just the last key pressed
                        key_resp_8.rt = _key_resp_8_allKeys[-1].rt
                        key_resp_8.duration = _key_resp_8_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in confirm_continueComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "confirm_continue" ---
            for thisComponent in confirm_continueComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('confirm_continue.stopped', globalClock.getTime())
            # Run 'End Routine' code from set_therm_2
            # Log intensity
            Ramp.addData('intensity', intensity)
            
            # Log wait_jitter
            Ramp.addData('wait1_dur', wait1jitter)
            
            # QUit if q pressed
            if 'q' in key_resp_8.keys :
                    Ramp.finished = 1
            # the Routine "confirm_continue" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "wait" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('wait.started', globalClock.getTime())
            text_3.setText('+')
            # keep track of which components have finished
            waitComponents = [text_3, text_14]
            for thisComponent in waitComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "wait" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_3* updates
                
                # if text_3 is starting this frame...
                if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_3.frameNStart = frameN  # exact frame index
                    text_3.tStart = t  # local t and not account for scr refresh
                    text_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    text_3.status = STARTED
                    text_3.setAutoDraw(True)
                
                # if text_3 is active this frame...
                if text_3.status == STARTED:
                    # update params
                    pass
                
                # if text_3 is stopping this frame...
                if text_3.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_3.tStartRefresh + wait1jitter-frameTolerance:
                        # keep track of stop time/frame for later
                        text_3.tStop = t  # not accounting for scr refresh
                        text_3.frameNStop = frameN  # exact frame index
                        # update status
                        text_3.status = FINISHED
                        text_3.setAutoDraw(False)
                
                # *text_14* updates
                
                # if text_14 is starting this frame...
                if text_14.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_14.frameNStart = frameN  # exact frame index
                    text_14.tStart = t  # local t and not account for scr refresh
                    text_14.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_14, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    text_14.status = STARTED
                    text_14.setAutoDraw(True)
                
                # if text_14 is active this frame...
                if text_14.status == STARTED:
                    # update params
                    text_14.setText(thermode_localisation
                    , log=False)
                
                # if text_14 is stopping this frame...
                if text_14.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_14.tStartRefresh + wait1jitter-frameTolerance:
                        # keep track of stop time/frame for later
                        text_14.tStop = t  # not accounting for scr refresh
                        text_14.frameNStop = frameN  # exact frame index
                        # update status
                        text_14.status = FINISHED
                        text_14.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in waitComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "wait" ---
            for thisComponent in waitComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('wait.stopped', globalClock.getTime())
            # the Routine "wait" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "therm_trig" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('therm_trig.started', globalClock.getTime())
            # Run 'Begin Routine' code from therm_deliver
            if fired == 0:
                fired = 1
                # Trigger thermode
                out_file = u'data/%s_%s/Calibration/%s_%s_%s%s' % (expInfo['participant'], data.getDateStr(format="%Y-%m-%d"), expInfo['participant'], expName, expInfo['date'], expInfo['participant'] + '_temp_trial_calib_ramp_' + str(series.thisN).zfill(2) + '_temp_' + str(intensity) + '.csv')
                stim_thread = threading.Thread(target=port_thermode.trigger_and_save_temp_rd,
                                                args=(out_file, 11500,))
                stim_thread.start()
            
            
            
            
            # keep track of which components have finished
            therm_trigComponents = [text_15]
            for thisComponent in therm_trigComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "therm_trig" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 12.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from therm_deliver
                # Draw polygon
                polygon_1.draw()
                
                
                
                # *text_15* updates
                
                # if text_15 is starting this frame...
                if text_15.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_15.frameNStart = frameN  # exact frame index
                    text_15.tStart = t  # local t and not account for scr refresh
                    text_15.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_15, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    text_15.status = STARTED
                    text_15.setAutoDraw(True)
                
                # if text_15 is active this frame...
                if text_15.status == STARTED:
                    # update params
                    text_15.setText(thermode_localisation
                    , log=False)
                
                # if text_15 is stopping this frame...
                if text_15.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_15.tStartRefresh + 12-frameTolerance:
                        # keep track of stop time/frame for later
                        text_15.tStop = t  # not accounting for scr refresh
                        text_15.frameNStop = frameN  # exact frame index
                        # update status
                        text_15.status = FINISHED
                        text_15.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in therm_trigComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "therm_trig" ---
            for thisComponent in therm_trigComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('therm_trig.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-12.000000)
            
            # --- Prepare to start Routine "rating_scale" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('rating_scale.started', globalClock.getTime())
            # Run 'Begin Routine' code from slidercode
            
            # Initialize rating scale
            ratingScale = visual.RatingScale(win=win, name='ratingScale', lineColor=(255, 255, 255), low=0, high=100, precision=100, size=1, tickMarks=None, tickHeight=0, scale=None, labels=['Aucune\ndouleur','Pire douleur\nimaginable'], marker=visual.Rect(win, width=0.01, height=0.1, lineColor='white', fillColor='white', units='norm'), markerColor=(255, 255, 255), markerStart=0.5, textColor='white', pos=(0, 0), stretch=2, showValue=None, showAccept=None, textSize=1.2)
            
            # Get a random initial posistion
            pos = np.random.randint(1, 10)
            ratingScale.setMarkerPos(pos)
            
            # Max time for rating scale
            respDisplay = ''
            maxResp = 5
            
            #key logger defaults
            last_len = 0
            key_list = []
            ratingScale.reset()
            # keep track of which components have finished
            rating_scaleComponents = [ratingScale, main_text, confirm_text, text_16]
            for thisComponent in rating_scaleComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "rating_scale" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from slidercode
                
                
                inc = 0.05
                wasmoved = 0
                
                while True:
                    keys = kb.getKeys(['m', 'n', 'space', 'escape'], waitRelease=False, clear=False)
                    if keys and not keys[-1].duration:
                        key = keys[-1].name
                        if 'n' == key:
                            pos -= inc
                            wasmoved = 1
                        if 'm' == key:
                            pos += inc
                            wasmoved = 1
                
                        if pos > 10:
                            pos = 10
                        elif pos < 0:
                            pos = 0
                            
                        # check for quit (typically the Esc key)
                        if "escape" ==  key:
                            core.quit()
                            
                        if 'space' == key:
                            core.wait(0.1)
                            continueRoutine=False
                            break
                
                    ratingScale.setMarkerPos(pos)
                    ratingScale.draw()
                    main_text.draw()
                    confirm_text.draw()
                    win.mouseVisible = False
                    win.flip()
                
                
                
                
                # *ratingScale* updates
                
                # if ratingScale is starting this frame...
                if ratingScale.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    ratingScale.frameNStart = frameN  # exact frame index
                    ratingScale.tStart = t  # local t and not account for scr refresh
                    ratingScale.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(ratingScale, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ratingScale.started')
                    # update status
                    ratingScale.status = STARTED
                    ratingScale.setAutoDraw(True)
                continueRoutine &= ratingScale.noResponse  # a response ends the trial
                
                # *main_text* updates
                
                # if main_text is starting this frame...
                if main_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    main_text.frameNStart = frameN  # exact frame index
                    main_text.tStart = t  # local t and not account for scr refresh
                    main_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(main_text, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    main_text.status = STARTED
                    main_text.setAutoDraw(True)
                
                # if main_text is active this frame...
                if main_text.status == STARTED:
                    # update params
                    pass
                
                # *confirm_text* updates
                
                # if confirm_text is starting this frame...
                if confirm_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    confirm_text.frameNStart = frameN  # exact frame index
                    confirm_text.tStart = t  # local t and not account for scr refresh
                    confirm_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(confirm_text, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    confirm_text.status = STARTED
                    confirm_text.setAutoDraw(True)
                
                # if confirm_text is active this frame...
                if confirm_text.status == STARTED:
                    # update params
                    pass
                
                # *text_16* updates
                
                # if text_16 is starting this frame...
                if text_16.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_16.frameNStart = frameN  # exact frame index
                    text_16.tStart = t  # local t and not account for scr refresh
                    text_16.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_16, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    text_16.status = STARTED
                    text_16.setAutoDraw(True)
                
                # if text_16 is active this frame...
                if text_16.status == STARTED:
                    # update params
                    text_16.setText(thermode_localisation
                    , log=False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in rating_scaleComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "rating_scale" ---
            for thisComponent in rating_scaleComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('rating_scale.stopped', globalClock.getTime())
            # Run 'End Routine' code from slidercode
            # Add data
            thisExp.addData('rating', ratingScale.getRating())
            # store data for Ramp (TrialHandler)
            Ramp.addData('ratingScale.response', ratingScale.getRating())
            Ramp.addData('ratingScale.rt', ratingScale.getRT())
            # the Routine "rating_scale" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "press_when_ready" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('press_when_ready.started', globalClock.getTime())
            key_resp_3.keys = []
            key_resp_3.rt = []
            _key_resp_3_allKeys = []
            # keep track of which components have finished
            press_when_readyComponents = [text_2, key_resp_3, text_17]
            for thisComponent in press_when_readyComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "press_when_ready" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_2* updates
                
                # if text_2 is starting this frame...
                if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_2.frameNStart = frameN  # exact frame index
                    text_2.tStart = t  # local t and not account for scr refresh
                    text_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_2.started')
                    # update status
                    text_2.status = STARTED
                    text_2.setAutoDraw(True)
                
                # if text_2 is active this frame...
                if text_2.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_3* updates
                waitOnFlip = False
                
                # if key_resp_3 is starting this frame...
                if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_3.frameNStart = frameN  # exact frame index
                    key_resp_3.tStart = t  # local t and not account for scr refresh
                    key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_3.started')
                    # update status
                    key_resp_3.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_3.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_3.getKeys(keyList=['y', 'n'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_3_allKeys.extend(theseKeys)
                    if len(_key_resp_3_allKeys):
                        key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                        key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                        key_resp_3.duration = _key_resp_3_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *text_17* updates
                
                # if text_17 is starting this frame...
                if text_17.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_17.frameNStart = frameN  # exact frame index
                    text_17.tStart = t  # local t and not account for scr refresh
                    text_17.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_17, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    text_17.status = STARTED
                    text_17.setAutoDraw(True)
                
                # if text_17 is active this frame...
                if text_17.status == STARTED:
                    # update params
                    text_17.setText(thermode_localisation
                    , log=False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in press_when_readyComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "press_when_ready" ---
            for thisComponent in press_when_readyComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('press_when_ready.stopped', globalClock.getTime())
            # Run 'End Routine' code from log_code
            series_trial.append(series.thisN)
            ramp_trial.append(Ramp.thisN)
            intensities_list.append(intensity)
            fired_list.append(fired)
            ratings_list.append(ratingScale.getRating())
            
            
            
            # QUit if q pressed
            if 'n' in key_resp_3.keys :
                    Ramp.finished = 1
            
            #Next block if intensity therm = 50c
            if therm_intensity[curr_item] == 50 :
                Ramp.finished = 1
            
            # Reset thermode counter if needed
            if curr_item_loc >= 3 :
                curr_item_loc = -1
                    
            
              
            
            
            # check responses
            if key_resp_3.keys in ['', [], None]:  # No response was made
                key_resp_3.keys = None
            Ramp.addData('key_resp_3.keys',key_resp_3.keys)
            if key_resp_3.keys != None:  # we had a response
                Ramp.addData('key_resp_3.rt', key_resp_3.rt)
                Ramp.addData('key_resp_3.duration', key_resp_3.duration)
            # the Routine "press_when_ready" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 10 repeats of 'Ramp'
        
        
        # --- Prepare to start Routine "bloc_terminé" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('bloc_terminé.started', globalClock.getTime())
        # keep track of which components have finished
        bloc_terminéComponents = [block_done]
        for thisComponent in bloc_terminéComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "bloc_terminé" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 4.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *block_done* updates
            
            # if block_done is starting this frame...
            if block_done.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                block_done.frameNStart = frameN  # exact frame index
                block_done.tStart = t  # local t and not account for scr refresh
                block_done.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(block_done, 'tStartRefresh')  # time at next scr refresh
                # update status
                block_done.status = STARTED
                block_done.setAutoDraw(True)
            
            # if block_done is active this frame...
            if block_done.status == STARTED:
                # update params
                pass
            
            # if block_done is stopping this frame...
            if block_done.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > block_done.tStartRefresh + 4.0-frameTolerance:
                    # keep track of stop time/frame for later
                    block_done.tStop = t  # not accounting for scr refresh
                    block_done.frameNStop = frameN  # exact frame index
                    # update status
                    block_done.status = FINISHED
                    block_done.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in bloc_terminéComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "bloc_terminé" ---
        for thisComponent in bloc_terminéComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('bloc_terminé.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-4.000000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed expInfo['nramps'] repeats of 'series'
    
    
    # --- Prepare to start Routine "compile_ramps" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('compile_ramps.started', globalClock.getTime())
    # Run 'Begin Routine' code from compile
    # Get data in a csv file
    ramp_data = {'series': series_trial,
                  'trial' : ramp_trial,
                  'intensity': intensities_list,
                  'rating': ratings_list,
                  'fired': fired_list}
    
    print('ramp_data is :', ramp_data)
    
    ramp_data = pd.DataFrame(data=ramp_data)
    
    # Save ramp data before next phase
    filenameramp = _thisDir + os.sep + u'data/%s_%s/Calibration/%s_%s_%s_%s' % (expInfo['participant'], data.getDateStr(format="%Y-%m-%d"), expInfo['participant'], expName, expInfo['date'], 'ramp_ratings.csv' )
    ramp_data.to_csv(filenameramp)
    
    # Use only last run to get threshold and tolerance
    ramp_data_last = ramp_data[ramp_data.series==2]
    
    # Remove misfires
    ramp_data_last = ramp_data_last[ramp_data_last['fired'] == 1]
    
    # Get tolerance level 
    tolerance = np.max(ramp_data_last.intensity)
    
    # Get min intensity above threshold
    ramp_data_above = ramp_data_last[ramp_data_last['rating'] > 0]
    threshold = np.min(ramp_data_above['intensity'])
    
    ramp_data['tolerance'] = tolerance
    ramp_data['threshold'] = threshold
    #ramp_data.to_csv(filenameramp)
    
    # Generate  intensities from low to tolerance
    rand_intensities = np.around(np.linspace(threshold, tolerance, 14), 1)
    np.random.shuffle(rand_intensities)
    
    
    # keep track of which components have finished
    compile_rampsComponents = [premiere_partie]
    for thisComponent in compile_rampsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "compile_ramps" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *premiere_partie* updates
        
        # if premiere_partie is starting this frame...
        if premiere_partie.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            premiere_partie.frameNStart = frameN  # exact frame index
            premiere_partie.tStart = t  # local t and not account for scr refresh
            premiere_partie.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(premiere_partie, 'tStartRefresh')  # time at next scr refresh
            # update status
            premiere_partie.status = STARTED
            premiere_partie.setAutoDraw(True)
        
        # if premiere_partie is active this frame...
        if premiere_partie.status == STARTED:
            # update params
            pass
        
        # if premiere_partie is stopping this frame...
        if premiere_partie.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > premiere_partie.tStartRefresh + 5.0-frameTolerance:
                # keep track of stop time/frame for later
                premiere_partie.tStop = t  # not accounting for scr refresh
                premiere_partie.frameNStop = frameN  # exact frame index
                # update status
                premiere_partie.status = FINISHED
                premiere_partie.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in compile_rampsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "compile_ramps" ---
    for thisComponent in compile_rampsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('compile_ramps.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    
    # --- Prepare to start Routine "deuxieme_start" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('deuxieme_start.started', globalClock.getTime())
    key_resp_10.keys = []
    key_resp_10.rt = []
    _key_resp_10_allKeys = []
    # Run 'Begin Routine' code from load_csv
    
    # Get intensities
    intensity_list = rand_intensities
    
    # keep track of which components have finished
    deuxieme_startComponents = [debut_deuxieme, key_resp_10]
    for thisComponent in deuxieme_startComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "deuxieme_start" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *debut_deuxieme* updates
        
        # if debut_deuxieme is starting this frame...
        if debut_deuxieme.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            debut_deuxieme.frameNStart = frameN  # exact frame index
            debut_deuxieme.tStart = t  # local t and not account for scr refresh
            debut_deuxieme.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(debut_deuxieme, 'tStartRefresh')  # time at next scr refresh
            # update status
            debut_deuxieme.status = STARTED
            debut_deuxieme.setAutoDraw(True)
        
        # if debut_deuxieme is active this frame...
        if debut_deuxieme.status == STARTED:
            # update params
            pass
        
        # *key_resp_10* updates
        waitOnFlip = False
        
        # if key_resp_10 is starting this frame...
        if key_resp_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_10.frameNStart = frameN  # exact frame index
            key_resp_10.tStart = t  # local t and not account for scr refresh
            key_resp_10.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_10, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_10.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_10.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_10.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_10.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_10.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_10_allKeys.extend(theseKeys)
            if len(_key_resp_10_allKeys):
                key_resp_10.keys = _key_resp_10_allKeys[-1].name  # just the last key pressed
                key_resp_10.rt = _key_resp_10_allKeys[-1].rt
                key_resp_10.duration = _key_resp_10_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in deuxieme_startComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "deuxieme_start" ---
    for thisComponent in deuxieme_startComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('deuxieme_start.stopped', globalClock.getTime())
    # the Routine "deuxieme_start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=len(intensity_list), method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "press_when_ready2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('press_when_ready2.started', globalClock.getTime())
        key_resp_7.keys = []
        key_resp_7.rt = []
        _key_resp_7_allKeys = []
        # Run 'Begin Routine' code from code_part_2
        # Jitter waiting
        ramp_phase = 0
        wait1jitter = np.random.choice([2000, 2100, 2200, 2300, 2400, 2500])
        wait1jitter = wait1jitter/1000
        # Log intensity
        Ramp.addData('wait1_dur', wait1jitter)
        
                
        # Get the intensity
        curr_item_scd += 1
        valeur_temperature = intensity_list[curr_item_scd]
        
        fired = 0
        
        #Set temp & ramp thermode
        curr_temp = intensity_list[curr_item_scd]
        rise_curr = np.round((curr_temp - baseline)/rise_time)
        port_thermode.set_stim(target=curr_temp, rise_rate=rise_curr, return_rate=rise_curr,
                    dur_ms=11750,
                    dur_mode='fixed_total',
                    surfaces=[1, 2, 3, 4, 5])
        
        #Thermode localisation
        curr_item_loc += 1
        thermode_localisation = 'T: ' + str(curr_item_loc) 
        # keep track of which components have finished
        press_when_ready2Components = [text_9, key_resp_7, text_18]
        for thisComponent in press_when_ready2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "press_when_ready2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_9* updates
            
            # if text_9 is starting this frame...
            if text_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_9.frameNStart = frameN  # exact frame index
                text_9.tStart = t  # local t and not account for scr refresh
                text_9.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_9, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_9.started')
                # update status
                text_9.status = STARTED
                text_9.setAutoDraw(True)
            
            # if text_9 is active this frame...
            if text_9.status == STARTED:
                # update params
                pass
            
            # *key_resp_7* updates
            waitOnFlip = False
            
            # if key_resp_7 is starting this frame...
            if key_resp_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_7.frameNStart = frameN  # exact frame index
                key_resp_7.tStart = t  # local t and not account for scr refresh
                key_resp_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_7, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_resp_7.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_7.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_7.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_7.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_7.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_7_allKeys.extend(theseKeys)
                if len(_key_resp_7_allKeys):
                    key_resp_7.keys = _key_resp_7_allKeys[-1].name  # just the last key pressed
                    key_resp_7.rt = _key_resp_7_allKeys[-1].rt
                    key_resp_7.duration = _key_resp_7_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *text_18* updates
            
            # if text_18 is starting this frame...
            if text_18.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_18.frameNStart = frameN  # exact frame index
                text_18.tStart = t  # local t and not account for scr refresh
                text_18.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_18, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_18.status = STARTED
                text_18.setAutoDraw(True)
            
            # if text_18 is active this frame...
            if text_18.status == STARTED:
                # update params
                text_18.setText(thermode_localisation
                , log=False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in press_when_ready2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "press_when_ready2" ---
        for thisComponent in press_when_ready2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('press_when_ready2.stopped', globalClock.getTime())
        # the Routine "press_when_ready2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "wait_2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('wait_2.started', globalClock.getTime())
        wait_text2.setText('+')
        # keep track of which components have finished
        wait_2Components = [wait_text2, text_19]
        for thisComponent in wait_2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "wait_2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *wait_text2* updates
            
            # if wait_text2 is starting this frame...
            if wait_text2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                wait_text2.frameNStart = frameN  # exact frame index
                wait_text2.tStart = t  # local t and not account for scr refresh
                wait_text2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(wait_text2, 'tStartRefresh')  # time at next scr refresh
                # update status
                wait_text2.status = STARTED
                wait_text2.setAutoDraw(True)
            
            # if wait_text2 is active this frame...
            if wait_text2.status == STARTED:
                # update params
                pass
            
            # if wait_text2 is stopping this frame...
            if wait_text2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > wait_text2.tStartRefresh + wait1jitter-frameTolerance:
                    # keep track of stop time/frame for later
                    wait_text2.tStop = t  # not accounting for scr refresh
                    wait_text2.frameNStop = frameN  # exact frame index
                    # update status
                    wait_text2.status = FINISHED
                    wait_text2.setAutoDraw(False)
            
            # *text_19* updates
            
            # if text_19 is starting this frame...
            if text_19.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_19.frameNStart = frameN  # exact frame index
                text_19.tStart = t  # local t and not account for scr refresh
                text_19.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_19, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_19.status = STARTED
                text_19.setAutoDraw(True)
            
            # if text_19 is active this frame...
            if text_19.status == STARTED:
                # update params
                text_19.setText(thermode_localisation
                , log=False)
            
            # if text_19 is stopping this frame...
            if text_19.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_19.tStartRefresh + wait1jitter-frameTolerance:
                    # keep track of stop time/frame for later
                    text_19.tStop = t  # not accounting for scr refresh
                    text_19.frameNStop = frameN  # exact frame index
                    # update status
                    text_19.status = FINISHED
                    text_19.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in wait_2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "wait_2" ---
        for thisComponent in wait_2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('wait_2.stopped', globalClock.getTime())
        # the Routine "wait_2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "therm_trig2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('therm_trig2.started', globalClock.getTime())
        # Run 'Begin Routine' code from therm_deliver_2
        if fired == 0:
            fired = 1
            out_file = u'data/%s_%s/Calibration/%s_%s_%s%s' % (expInfo['participant'], data.getDateStr(format="%Y-%m-%d"), expInfo['participant'], expName, expInfo['date'], expInfo['participant'] + '_temp_trial_calib_random_' + str(trials.thisN).zfill(2) + '_temp_' + str(curr_temp) + '.csv')
            stim_thread = threading.Thread(target=port_thermode.trigger_and_save_temp_rd,
                                            args=(out_file, 11500,))
            stim_thread.start()
        
        # keep track of which components have finished
        therm_trig2Components = [text_20]
        for thisComponent in therm_trig2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "therm_trig2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 12.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from therm_deliver_2
            polygon_1.draw()
            
            # *text_20* updates
            
            # if text_20 is starting this frame...
            if text_20.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_20.frameNStart = frameN  # exact frame index
                text_20.tStart = t  # local t and not account for scr refresh
                text_20.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_20, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_20.status = STARTED
                text_20.setAutoDraw(True)
            
            # if text_20 is active this frame...
            if text_20.status == STARTED:
                # update params
                text_20.setText(thermode_localisation
                , log=False)
            
            # if text_20 is stopping this frame...
            if text_20.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_20.tStartRefresh + 12-frameTolerance:
                    # keep track of stop time/frame for later
                    text_20.tStop = t  # not accounting for scr refresh
                    text_20.frameNStop = frameN  # exact frame index
                    # update status
                    text_20.status = FINISHED
                    text_20.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in therm_trig2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "therm_trig2" ---
        for thisComponent in therm_trig2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('therm_trig2.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-12.000000)
        
        # --- Prepare to start Routine "rating_scale2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('rating_scale2.started', globalClock.getTime())
        keyResp_2.keys = []
        keyResp_2.rt = []
        _keyResp_2_allKeys = []
        # Run 'Begin Routine' code from slidercode_2
        
        ratingScale_2 = visual.RatingScale(win=win, name='ratingScale', lineColor=(255, 255, 255), low=0, high=100, precision=100, size=1, tickMarks=None, tickHeight=0, scale=None, labels=['Aucune\ndouleur','Pire douleur\nimaginable'], marker=visual.Rect(win, width=0.01, height=0.1, lineColor='white', fillColor='white', units='norm'), markerColor=(255, 255, 255), markerStart=0.5, textColor='white', pos=(0, 0), stretch=2, showValue=None, showAccept=None, textSize=1.2)
        
        pos = np.random.randint(1, 10)
        ratingScale_2.setMarkerPos(pos)
        
        
        
        ratingScale_2.reset()
        # keep track of which components have finished
        rating_scale2Components = [keyResp_2, ratingScale_2, main_text_2, confirm_text_2, text_21]
        for thisComponent in rating_scale2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "rating_scale2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *keyResp_2* updates
            waitOnFlip = False
            
            # if keyResp_2 is starting this frame...
            if keyResp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                keyResp_2.frameNStart = frameN  # exact frame index
                keyResp_2.tStart = t  # local t and not account for scr refresh
                keyResp_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(keyResp_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                keyResp_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(keyResp_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(keyResp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if keyResp_2.status == STARTED and not waitOnFlip:
                theseKeys = keyResp_2.getKeys(keyList=['a','1','2','3','4','5','6','7','8','9','0','return','backspace'], ignoreKeys=["escape"], waitRelease=False)
                _keyResp_2_allKeys.extend(theseKeys)
                if len(_keyResp_2_allKeys):
                    keyResp_2.keys = [key.name for key in _keyResp_2_allKeys]  # storing all keys
                    keyResp_2.rt = [key.rt for key in _keyResp_2_allKeys]
                    keyResp_2.duration = [key.duration for key in _keyResp_2_allKeys]
            # Run 'Each Frame' code from slidercode_2
            
            
            inc = 0.05
            wasmoved = 0
            
            while True:
                keys = kb.getKeys(['m', 'n', 'space', 'escape'], waitRelease=False, clear=False)
                if keys and not keys[-1].duration:
                    key = keys[-1].name
                    if 'n' == key:
                        pos -= inc
                        wasmoved = 1
                    if 'm' == key:
                        pos += inc
                        wasmoved = 1
            
                    if pos > 10:
                        pos = 10
                    elif pos < 0:
                        pos = 0
                        
                    # check for quit (typically the Esc key)
                    if "escape" ==  key:
                        core.quit()
                        
                    if 'space' == key:
                        print(ratingScale_2.getRating())
                        core.wait(0.1)
                        continueRoutine=False
                        break
            
                #print(kb.state)
                ratingScale_2.setMarkerPos(pos)
                ratingScale_2.draw()
                main_text.draw()
                confirm_text.draw()
                win.mouseVisible = False
                win.flip()
            
            
            
            
            # *ratingScale_2* updates
            
            # if ratingScale_2 is starting this frame...
            if ratingScale_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                ratingScale_2.frameNStart = frameN  # exact frame index
                ratingScale_2.tStart = t  # local t and not account for scr refresh
                ratingScale_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ratingScale_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ratingScale_2.started')
                # update status
                ratingScale_2.status = STARTED
                ratingScale_2.setAutoDraw(True)
            continueRoutine &= ratingScale_2.noResponse  # a response ends the trial
            
            # *main_text_2* updates
            
            # if main_text_2 is starting this frame...
            if main_text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                main_text_2.frameNStart = frameN  # exact frame index
                main_text_2.tStart = t  # local t and not account for scr refresh
                main_text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(main_text_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                main_text_2.status = STARTED
                main_text_2.setAutoDraw(True)
            
            # if main_text_2 is active this frame...
            if main_text_2.status == STARTED:
                # update params
                pass
            
            # *confirm_text_2* updates
            
            # if confirm_text_2 is starting this frame...
            if confirm_text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                confirm_text_2.frameNStart = frameN  # exact frame index
                confirm_text_2.tStart = t  # local t and not account for scr refresh
                confirm_text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(confirm_text_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                confirm_text_2.status = STARTED
                confirm_text_2.setAutoDraw(True)
            
            # if confirm_text_2 is active this frame...
            if confirm_text_2.status == STARTED:
                # update params
                pass
            
            # *text_21* updates
            
            # if text_21 is starting this frame...
            if text_21.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_21.frameNStart = frameN  # exact frame index
                text_21.tStart = t  # local t and not account for scr refresh
                text_21.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_21, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_21.status = STARTED
                text_21.setAutoDraw(True)
            
            # if text_21 is active this frame...
            if text_21.status == STARTED:
                # update params
                text_21.setText(thermode_localisation
                , log=False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in rating_scale2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "rating_scale2" ---
        for thisComponent in rating_scale2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('rating_scale2.stopped', globalClock.getTime())
        # check responses
        if keyResp_2.keys in ['', [], None]:  # No response was made
            keyResp_2.keys = None
        trials.addData('keyResp_2.keys',keyResp_2.keys)
        if keyResp_2.keys != None:  # we had a response
            trials.addData('keyResp_2.rt', keyResp_2.rt)
            trials.addData('keyResp_2.duration', keyResp_2.duration)
        # Run 'End Routine' code from slidercode_2
        Ramp.addData('therm_value', valeur_temperature)
        Ramp.addData('rating', ratingScale_2.getRating())
        ratings_random.append(ratingScale_2.getRating())
        
        if curr_item_loc >= 3 :
            curr_item_loc = -1
        # store data for trials (TrialHandler)
        trials.addData('ratingScale_2.response', ratingScale_2.getRating())
        trials.addData('ratingScale_2.rt', ratingScale_2.getRT())
        # the Routine "rating_scale2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed len(intensity_list) repeats of 'trials'
    
    
    # --- Prepare to start Routine "thank_you_output_code" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('thank_you_output_code.started', globalClock.getTime())
    # Run 'Begin Routine' code from choose_stim_2
    # Get data in a csv file
    ramp_data = {'intensity': intensity_list,
                  'rating': ratings_random}
    
    # Get tolerance level and offset a bit
    tolerance = np.max(intensity_list)
    
    # Création d'un DataFrame à partir du dictionnaire 'ramp_data'
    ramp_df = pd.DataFrame(ramp_data)
    print('ramp_df is : ', ramp_df)
    
    # Sélection des lignes avec un 'rating' positif
    ramp_data_above = ramp_df[ramp_df['rating'] > 0]
    
    # Obtenir le minimum d'intensité parmi les valeurs au-dessus du seuil
    threshold = np.min(ramp_data_above['intensity'])
    
    
    
    # Get min intensity above threshold
    #ramp_data_above = ramp_data[ramp_data['rating'] > 0]
    #threshold = np.min(ramp_data_above['intensity'])
    
    # Filtrer les données avec un rating positif
    #ramp_data_above = {
        #'intensity': [intensity for intensity, rating in zip(valeur_temperature, ratings_random) if rating > 0],
        #'rating': [rating for rating in ratings_random if rating > 0]}
    # keep track of which components have finished
    thank_you_output_codeComponents = [text_13]
    for thisComponent in thank_you_output_codeComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "thank_you_output_code" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_13* updates
        
        # if text_13 is starting this frame...
        if text_13.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_13.frameNStart = frameN  # exact frame index
            text_13.tStart = t  # local t and not account for scr refresh
            text_13.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_13, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_13.status = STARTED
            text_13.setAutoDraw(True)
        
        # if text_13 is active this frame...
        if text_13.status == STARTED:
            # update params
            pass
        
        # if text_13 is stopping this frame...
        if text_13.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_13.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                text_13.tStop = t  # not accounting for scr refresh
                text_13.frameNStop = frameN  # exact frame index
                # update status
                text_13.status = FINISHED
                text_13.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in thank_you_output_codeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "thank_you_output_code" ---
    for thisComponent in thank_you_output_codeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('thank_you_output_code.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    # Run 'End Experiment' code from choose_stim_2
    
    xdata = [i for i in intensity_list]
    
    target_rating = [0.3, 0.6]
    target_rating = [c*100 for c in target_rating]
    ratings_random = [(c/np.max(ratings_random))*100 for c in ratings_random]
    ydata = ratings_random
    intensities = np.arange(threshold, tolerance+0.1, 0.1)
    
    rate_data = pd.DataFrame()
    rate_data['ydata'] = ydata
    rate_data['xdata'] = xdata
    
    filenameramp_png = _thisDir + os.sep + u'data/%s_%s/Calibration/%s_%s_%s_%s' % (expInfo['participant'], data.getDateStr(format="%Y-%m-%d"), expInfo['participant'], expName, expInfo['date'], 'fitted_values.png' )
    filenameramp_csv = _thisDir + os.sep + u'data/%s_%s/Calibration/%s_%s_%s_%s' % (expInfo['participant'], data.getDateStr(format="%Y-%m-%d"), expInfo['participant'], expName, expInfo['date'], 'fitted_values.csv' )
    
    
    def fit_model(func_type, params, levels):
        return {
            #Scale down by 0.1 to prevent overflow error
            "linear" : params[0] + params[1] * levels,
            "hyper"  : params[0] + params[1]/(params[2] + levels),
            "para"   : params[0] + params[1] * levels + params[2] * (levels**2),
            "expo"   : params[0] + params[1]*(levels**params[2])
            }[func_type] 
    
    def get_model_sse(params, rate_data, func):
    
        # mean center to remove intercept
        fitted = fit_model(func, params, rate_data['xdata'])
        sse = np.sum((rate_data['ydata'] - fitted)**2)
    
        return sse
    
    results = dict()
    models = ['linear', 'hyper', 'para', 'expo']
    for func in models:
        results[func] = dict()
        res = minimize(fun=get_model_sse, x0=[1, 1, 1],
                        args=(rate_data, func),
                        method='SLSQP',
                        options={'maxiter': 100})
    
        results[func]['fitted'] = fit_model(func, res['x'], intensities)
        results[func]['sse'] = res['fun']
        results[func]['params'] = res['x']
        results[func]['stim_val'] = []
        results[func]['fitted_val'] = []
    
        for t in target_rating:
            loc = np.argmin(np.abs(results[func]['fitted'] - t))
            results[func]['stim_val'].append(intensities[loc])
            results[func]['fitted_val'].append(results[func]['fitted'] [loc])
        # Save at each model in case crash
        pd.DataFrame(results).to_csv(filenameramp_csv)
        win_model = models[np.argmin([results[c]['sse'] for c in list(results.keys())])]
        results_temp = results.copy()
        results_temp['win_model'] = win_model
        results_temp['target_ratings'] = str(target_rating)
        results_temp['selected_intensities'] = str(results[win_model]['stim_val'])
        results_temp['extrapolated_ratings'] = str(results[win_model]['fitted_val'])
        pd.DataFrame(results_temp).to_csv(filenameramp_csv)
        # Plot
        calib_plot = plt.figure(figsize=(6, 5))
        plt.scatter(xdata, ydata, label='Actual ratings')
        plt.plot(intensities, results[win_model]['fitted'], label='Best fit', color='orange')
        plt.scatter(results[win_model]['stim_val'], results[win_model]['fitted_val'], label=win_model
                    + ' fitted values',
                    alpha=0.9, marker='^', s=60, color='orange')
        plt.plot(intensities, results['linear']['fitted'], label='Linear fit', color='green')
        plt.scatter(results['linear']['stim_val'], results['linear']['fitted_val'], label='Linearly spaced values',
                    alpha=0.5, color='green')
        plt.ylim(-1, 105)
        # plt.title(expInfo['participant'])
        plt.xlabel('Intensity (Celsius)')
        plt.ylabel('Rating (% max rating)')
        plt.legend()
        plt.savefig(filenameramp_png)
    
    win_model = models[np.argmin([results[c]['sse'] for c in list(results.keys())])]
    
    results['win_model'] = win_model
    results['target_ratings'] = str(target_rating)
    results['selected_intensities'] = str(results[win_model]['stim_val'])
    results['extrapolated_ratings'] = str(results[win_model]['fitted_val'])
    
    # Save final
    pd.DataFrame(results).to_csv(filenameramp_csv)
    # Plot
    calib_plot = plt.figure(figsize=(6, 5))
    plt.scatter(xdata, ydata, label='Actual ratings')
    plt.plot(intensities, results[win_model]['fitted'], label='Best fit', color='orange')
    plt.scatter(results[win_model]['stim_val'], results[win_model]['fitted_val'], label=win_model
                + ' fitted values',
                alpha=0.9, marker='^', s=60, color='orange')
    plt.plot(intensities, results['linear']['fitted'], label='Linear fit', color='green')
    plt.scatter(results['linear']['stim_val'], results['linear']['fitted_val'], label='Linearly spaced values',
                alpha=0.5, color='green')
    plt.ylim(-1, 105)
    # plt.title(expInfo['participant'])
    plt.xlabel('Intensity (Celsius)')
    plt.ylabel('Rating (% max rating)')
    plt.legend()
    plt.savefig(filenameramp_png)
    
    
    
    
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
