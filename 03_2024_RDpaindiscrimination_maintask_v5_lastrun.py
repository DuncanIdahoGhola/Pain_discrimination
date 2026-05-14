#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on May 13, 2026, at 15:34
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
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
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

# Run 'Before Experiment' code from Clock
import serial
timer = core.Clock() 
# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = '2024_RDpaindiscrimination_maintask'  # from the Builder filename that created this script
expInfo = {
    'participant': 'sub-000',
    'temp_flat': '',
    'temp_active': '',
    'temp_pic_set': '',
    'com_thermode': 'COM3',
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
    filename = u'data/%s_%s/%s_%s_%s' % (expInfo['participant'], data.getDateStr(format="%Y-%m-%d"), expInfo['participant'], expName, expInfo['date'] )
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\test\\Desktop\\RD_2024\\03_2024_RDpaindiscrimination_maintask_v5_lastrun.py',
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
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
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
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
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
    
    # --- Initialize components for Routine "Set_up" ---
    text_5 = visual.TextStim(win=win, name='text_5',
        text="Bienvenue!\n\nAppuyer sur 'p' lorsque vous serez prêt",
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_3 = keyboard.Keyboard()
    # Run 'Begin Experiment' code from init_thermode
    # EEG
    import struct
    import time
    import random
    import pandas as pd
    import threading
    
    
    # Thermode
    from pytcsii import tcsii_serial
    port_thermode = tcsii_serial(str(expInfo['com_thermode']))
    port_thermode.set_baseline(38)
    
    
    baseline = 38
    rise_time = .75
    
    temp_flat = float(expInfo['temp_flat'])
    temp_active = float(expInfo['temp_active'])
    temp_pic_set = float(expInfo['temp_pic_set'])
    
    rise_flat = np.round((temp_flat - baseline)/rise_time, 1)
    rise_active = np.round((temp_active - baseline)/rise_time, 1)
    
    part = int(expInfo['participant'][4:])-1
    
    # Part pair ou impair : Sélection couleur
    col1 = [1, 0.5059, -1]
    col2 = [-1, 0.38, 0.88]
    col3 = [-0.5, -0.5, -0.5]
    if part % 2 == 0:
        #Triangle bleu, carré jaune
        color_inactif_trig = col1
        color_actif_carre = col2
        #texte oui-non
        loca_textl = 'OUI              '
        loca_textr = '              NON'
        detectl = 1
        detectr = 0
    
        print('impair')
    else:
        #Triangle jaune, carrée bleu
        color_inactif_trig = col2
        color_actif_carre = col1
        print('pair')
        #texte oui-non
        loca_textl = 'NON              '
        loca_textr = '              OUI'
        detectl = 0
        detectr = 1
    
    
    #Polygon color
    polygon_1 = visual.ShapeStim(
        win=win, name='polygon',
        size=(0.5, 0.5), vertices='triangle',
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor= color_inactif_trig, fillColor= color_inactif_trig,
        opacity=1.0, depth=-1.0, interpolate=True)
    polygon_2 = visual.Rect(
        win=win, name='polygon_2',
        width=(0.5, 0.5)[0], height=(0.5, 0.5)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor= color_actif_carre, fillColor= color_actif_carre,
        opacity=None, depth=-2.0, interpolate=True)
    
    #Black polygon
    black = [-1.0000, -1.0000, -1.0000]
        
    polygon_1_black = visual.ShapeStim(
        win=win, name='polygon_1_black',
        size=(0.4, 0.4), vertices='triangle',
        ori=0.0, pos=(0, -.025), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor= col3, fillColor= col3,
        opacity=1.0, depth=-1.0, interpolate=True)
    
    polygon_2_black = visual.Rect(
        win=win, name='polygon_2_black',
        width=(0.4, 0.4)[0], height=(0.4, 0.4)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor= col3, fillColor= col3,
        opacity=None, depth=-2.0, interpolate=True)
    
    
    # Get file for randomisation
    liste_test = pd.read_csv('RD_discrimination_random_eval.csv')
    curr_item_test = -1
    
    liste_discri = pd.read_csv('RD_discrimination_random_discrimnation.csv')
    curr_item_discri = -1
    
    liste_eval = pd.read_csv('RD_discrimination_random_evalpla.csv')
    curr_item_eval = -1
    
    # Font size
    cross_size = 0.2
    loctherm_size = 0.02
    
    # --- Initialize components for Routine "electrode_check" ---
    text_4 = visual.TextStim(win=win, name='text_4',
        text="Nous allons vérifier si\nl'électrode fonctionne\n\n\nExpérimentateur.trice,\n appuyer sur 'p'",
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_5 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "instructions_eval" ---
    text_7 = visual.TextStim(win=win, name='text_7',
        text="Pour les prochains essais, vous devrez ÉVALUER l'intensité de la douleur ressentie. ",
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_6 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "pause_5_test" ---
    # Run 'Begin Experiment' code from get_file_eval
    
    
    
    text_2 = visual.TextStim(win=win, name='text_2',
        text='',
        font='Open Sans',
        pos=(-.70, -.45), height=loctherm_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    croix = visual.TextStim(win=win, name='croix',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=cross_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "Phase_Test" ---
    thermode_locali = visual.TextStim(win=win, name='thermode_locali',
        text='',
        font='Open Sans',
        pos=(-.70, -.45), height=loctherm_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "eval_douleur" ---
    # Run 'Begin Experiment' code from slidercode
    kb = keyboard.Keyboard()
    win.mouseVisible = False
    ratingScale = visual.RatingScale(win=win, name='ratingScale', lineColor=(255, 255, 255), low=0, high=1000, precision=1000, size=1, tickMarks=None, tickHeight=1, scale=None, labels=None, marker=None, markerColor=(255, 255, 255), markerStart=0.5, textColor=(255, 255, 255), pos=(0, 0), stretch=2, showValue=None, showAccept=None, textSize=1.2)
    main_text = visual.TextStim(win=win, name='main_text',
        text="\nVeuillez évaluer l'intensité de la stimulation que vous venez de recevoir.",
        font='Arial',
        pos=(0, 0.4), height=0.07, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-2.0);
    confirm_text = visual.TextStim(win=win, name='confirm_text',
        text='Appuyer sur le bouton du haut pour poursuivre.',
        font='Arial',
        pos=(0, -0.2), height=0.05, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "Fin_test" ---
    text = visual.TextStim(win=win, name='text',
        text="Électrode Ok\n\n2 Évaluations douleurs\n\nSuivi par \n\n4 Évaluations de discriminations\n\nRépété 4 fois\n\n\nAppuyer sur 'p' pour continuer",
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_4 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "instructions_eval" ---
    text_7 = visual.TextStim(win=win, name='text_7',
        text="Pour les prochains essais, vous devrez ÉVALUER l'intensité de la douleur ressentie. ",
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_6 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "pause_5_EVA" ---
    text_eva = visual.TextStim(win=win, name='text_eva',
        text='',
        font='Open Sans',
        pos=(-.70, -.45), height=loctherm_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    croix_2 = visual.TextStim(win=win, name='croix_2',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=cross_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "tâche_EVA" ---
    thermode_locali_EVA = visual.TextStim(win=win, name='thermode_locali_EVA',
        text='',
        font='Open Sans',
        pos=(-.70, -.45), height=loctherm_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "eval_douleur" ---
    # Run 'Begin Experiment' code from slidercode
    kb = keyboard.Keyboard()
    win.mouseVisible = False
    ratingScale = visual.RatingScale(win=win, name='ratingScale', lineColor=(255, 255, 255), low=0, high=1000, precision=1000, size=1, tickMarks=None, tickHeight=1, scale=None, labels=None, marker=None, markerColor=(255, 255, 255), markerStart=0.5, textColor=(255, 255, 255), pos=(0, 0), stretch=2, showValue=None, showAccept=None, textSize=1.2)
    main_text = visual.TextStim(win=win, name='main_text',
        text="\nVeuillez évaluer l'intensité de la stimulation que vous venez de recevoir.",
        font='Arial',
        pos=(0, 0.4), height=0.07, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-2.0);
    confirm_text = visual.TextStim(win=win, name='confirm_text',
        text='Appuyer sur le bouton du haut pour poursuivre.',
        font='Arial',
        pos=(0, -0.2), height=0.05, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "instructions_discrim" ---
    text_8 = visual.TextStim(win=win, name='text_8',
        text='Pour les prochains essais, vous devrez DÉTECTER un changement de température, en indiquant si le changement était présent ou non lorsque demandé.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_7 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "pause_5_pic" ---
    text_pic = visual.TextStim(win=win, name='text_pic',
        text='',
        font='Open Sans',
        pos=(-.70, -.45), height=loctherm_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    croix_3 = visual.TextStim(win=win, name='croix_3',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=cross_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "tâche_PIC" ---
    # Run 'Begin Experiment' code from trigger_pic
    
    # Get pic stim
    pic_str = str(int(np.round(temp_pic_set, 1)*10))
    temp_flat_str = str(int(np.round(temp_flat, 1)*10))
    
    
    
    thermode_locali_pic = visual.TextStim(win=win, name='thermode_locali_pic',
        text='',
        font='Open Sans',
        pos=(-.70, -.45), height=loctherm_size, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    discrimin_resp = keyboard.Keyboard()
    responsel = visual.TextStim(win=win, name='responsel',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.08, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    percu_or = visual.TextStim(win=win, name='percu_or',
        text='Avez-vous perçu un changement?',
        font='Open Sans',
        pos=(0, 0.3), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    responser = visual.TextStim(win=win, name='responser',
        text=loca_textr,
        font='Open Sans',
        pos=(0, 0), height=0.08, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    
    # --- Initialize components for Routine "break_entre_bloc" ---
    text_3 = visual.TextStim(win=win, name='text_3',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_2 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "merci" ---
    text_6 = visual.TextStim(win=win, name='text_6',
        text='Merci pour votre participation',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
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
    
    # --- Prepare to start Routine "Set_up" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Set_up.started', globalClock.getTime())
    key_resp_3.keys = []
    key_resp_3.rt = []
    _key_resp_3_allKeys = []
    # Run 'Begin Routine' code from init_thermode
    win.mouseVisible = False
    # keep track of which components have finished
    Set_upComponents = [text_5, key_resp_3]
    for thisComponent in Set_upComponents:
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
    
    # --- Run Routine "Set_up" ---
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
        
        # *key_resp_3* updates
        waitOnFlip = False
        
        # if key_resp_3 is starting this frame...
        if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_3.frameNStart = frameN  # exact frame index
            key_resp_3.tStart = t  # local t and not account for scr refresh
            key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_3.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_3.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_3_allKeys.extend(theseKeys)
            if len(_key_resp_3_allKeys):
                key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                key_resp_3.duration = _key_resp_3_allKeys[-1].duration
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
        for thisComponent in Set_upComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Set_up" ---
    for thisComponent in Set_upComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Set_up.stopped', globalClock.getTime())
    # the Routine "Set_up" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=2.0, method='random', 
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
        
        # --- Prepare to start Routine "electrode_check" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('electrode_check.started', globalClock.getTime())
        key_resp_5.keys = []
        key_resp_5.rt = []
        _key_resp_5_allKeys = []
        # keep track of which components have finished
        electrode_checkComponents = [text_4, key_resp_5]
        for thisComponent in electrode_checkComponents:
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
        
        # --- Run Routine "electrode_check" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_4* updates
            
            # if text_4 is starting this frame...
            if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_4.frameNStart = frameN  # exact frame index
                text_4.tStart = t  # local t and not account for scr refresh
                text_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_4.status = STARTED
                text_4.setAutoDraw(True)
            
            # if text_4 is active this frame...
            if text_4.status == STARTED:
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
            for thisComponent in electrode_checkComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "electrode_check" ---
        for thisComponent in electrode_checkComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('electrode_check.stopped', globalClock.getTime())
        # the Routine "electrode_check" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "instructions_eval" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('instructions_eval.started', globalClock.getTime())
        key_resp_6.keys = []
        key_resp_6.rt = []
        _key_resp_6_allKeys = []
        # keep track of which components have finished
        instructions_evalComponents = [text_7, key_resp_6]
        for thisComponent in instructions_evalComponents:
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
        
        # --- Run Routine "instructions_eval" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_7* updates
            
            # if text_7 is starting this frame...
            if text_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_7.frameNStart = frameN  # exact frame index
                text_7.tStart = t  # local t and not account for scr refresh
                text_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_7, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_7.status = STARTED
                text_7.setAutoDraw(True)
            
            # if text_7 is active this frame...
            if text_7.status == STARTED:
                # update params
                pass
            
            # *key_resp_6* updates
            waitOnFlip = False
            
            # if key_resp_6 is starting this frame...
            if key_resp_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_6.frameNStart = frameN  # exact frame index
                key_resp_6.tStart = t  # local t and not account for scr refresh
                key_resp_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_6, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_resp_6.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_6.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_6.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_6.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_6.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_6_allKeys.extend(theseKeys)
                if len(_key_resp_6_allKeys):
                    key_resp_6.keys = _key_resp_6_allKeys[-1].name  # just the last key pressed
                    key_resp_6.rt = _key_resp_6_allKeys[-1].rt
                    key_resp_6.duration = _key_resp_6_allKeys[-1].duration
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
            for thisComponent in instructions_evalComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "instructions_eval" ---
        for thisComponent in instructions_evalComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('instructions_eval.stopped', globalClock.getTime())
        # the Routine "instructions_eval" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        loop_test = data.TrialHandler(nReps=16.0, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='loop_test')
        thisExp.addLoop(loop_test)  # add the loop to the experiment
        thisLoop_test = loop_test.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_test.rgb)
        if thisLoop_test != None:
            for paramName in thisLoop_test:
                globals()[paramName] = thisLoop_test[paramName]
        
        for thisLoop_test in loop_test:
            currentLoop = loop_test
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
            # abbreviate parameter names if possible (e.g. rgb = thisLoop_test.rgb)
            if thisLoop_test != None:
                for paramName in thisLoop_test:
                    globals()[paramName] = thisLoop_test[paramName]
            
            # --- Prepare to start Routine "pause_5_test" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('pause_5_test.started', globalClock.getTime())
            # Run 'Begin Routine' code from get_file_eval
            # Get trials in psychopy
            curr_item_test += 1
            
            trial = 'trial_' + str(curr_item_test).zfill(2)
            print('test trial is ' + trial)
            
            cond, loc = liste_test.loc[part, trial].split('_')
            print('test cond and loc ' + cond + '   ' + loc)
            
            thermode_localisation = 'T: ' + str(loc) 
            
            
            # keep track of which components have finished
            pause_5_testComponents = [text_2, croix]
            for thisComponent in pause_5_testComponents:
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
            
            # --- Run Routine "pause_5_test" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 5.0:
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
                    # update status
                    text_2.status = STARTED
                    text_2.setAutoDraw(True)
                
                # if text_2 is active this frame...
                if text_2.status == STARTED:
                    # update params
                    text_2.setText(thermode_localisation
                    , log=False)
                
                # if text_2 is stopping this frame...
                if text_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_2.tStartRefresh + 5.0-frameTolerance:
                        # keep track of stop time/frame for later
                        text_2.tStop = t  # not accounting for scr refresh
                        text_2.frameNStop = frameN  # exact frame index
                        # update status
                        text_2.status = FINISHED
                        text_2.setAutoDraw(False)
                
                # *croix* updates
                
                # if croix is starting this frame...
                if croix.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    croix.frameNStart = frameN  # exact frame index
                    croix.tStart = t  # local t and not account for scr refresh
                    croix.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(croix, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    croix.status = STARTED
                    croix.setAutoDraw(True)
                
                # if croix is active this frame...
                if croix.status == STARTED:
                    # update params
                    pass
                
                # if croix is stopping this frame...
                if croix.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > croix.tStartRefresh + 5.0-frameTolerance:
                        # keep track of stop time/frame for later
                        croix.tStop = t  # not accounting for scr refresh
                        croix.frameNStop = frameN  # exact frame index
                        # update status
                        croix.status = FINISHED
                        croix.setAutoDraw(False)
                
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
                for thisComponent in pause_5_testComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "pause_5_test" ---
            for thisComponent in pause_5_testComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('pause_5_test.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-5.000000)
            
            # --- Prepare to start Routine "Phase_Test" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('Phase_Test.started', globalClock.getTime())
            # Run 'Begin Routine' code from Stim_selection
            
            # Disable temp profile
            port_thermode.port.write('Ue00000'.encode())
            
            dur_plateau1 = 4
            dur_plateau2 = 4.8
            print(cond)
            if cond == 'inactive' :
                port_thermode.set_stim(target=temp_flat, rise_rate=rise_flat, return_rate=rise_flat,
                            dur_ms=11500,
                            dur_mode='fixed_total',
                            surfaces=[1, 2, 3, 4, 5])
                temp_trial_sent = temp_flat
            
            elif cond == 'active' :
                port_thermode.set_stim(target=temp_active, rise_rate=rise_active, return_rate=rise_active,
                            dur_ms=11500,
                            dur_mode='fixed_total',
                            surfaces=[1, 2, 3, 4, 5])
                temp_trial_sent = temp_active
            print(temp_trial_sent)
            ## Trigger thermode in other thread
            out_file = u'data/%s_%s/%s_%s_%s%s' % (expInfo['participant'], data.getDateStr(format="%Y-%m-%d"), expInfo['participant'], expName, expInfo['date'], expInfo['participant'] + '_temp_trial_eval_cond_' + str(curr_item_test).zfill(2) + '.csv')
            stim_thread = threading.Thread(target=port_thermode.trigger_and_save_temp_rd,
                                            args=(out_file, 11500,))
            stim_thread.start()
            
            # keep track of which components have finished
            Phase_TestComponents = [thermode_locali]
            for thisComponent in Phase_TestComponents:
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
            
            # --- Run Routine "Phase_Test" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 12.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from Stim_selection
                if cond == 'active' :
                    polygon_1.draw()
                else :
                    polygon_2.draw()
                
                # *thermode_locali* updates
                
                # if thermode_locali is starting this frame...
                if thermode_locali.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    thermode_locali.frameNStart = frameN  # exact frame index
                    thermode_locali.tStart = t  # local t and not account for scr refresh
                    thermode_locali.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(thermode_locali, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'thermode_locali.started')
                    # update status
                    thermode_locali.status = STARTED
                    thermode_locali.setAutoDraw(True)
                
                # if thermode_locali is active this frame...
                if thermode_locali.status == STARTED:
                    # update params
                    thermode_locali.setText(thermode_localisation, log=False)
                
                # if thermode_locali is stopping this frame...
                if thermode_locali.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > thermode_locali.tStartRefresh + 12-frameTolerance:
                        # keep track of stop time/frame for later
                        thermode_locali.tStop = t  # not accounting for scr refresh
                        thermode_locali.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'thermode_locali.stopped')
                        # update status
                        thermode_locali.status = FINISHED
                        thermode_locali.setAutoDraw(False)
                
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
                for thisComponent in Phase_TestComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "Phase_Test" ---
            for thisComponent in Phase_TestComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('Phase_Test.stopped', globalClock.getTime())
            # Run 'End Routine' code from Stim_selection
            loop_test.addData('loca_thermode', loc)
            loop_test.addData('condition', cond)
            loop_test.addData('temp_trial_sent', temp_trial_sent)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-12.000000)
            
            # --- Prepare to start Routine "eval_douleur" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('eval_douleur.started', globalClock.getTime())
            # Run 'Begin Routine' code from slidercode
            
            ratingScale = visual.RatingScale(win=win, name='ratingScale', lineColor=(255, 255, 255), low=0, high=100, precision=100, size=1, tickMarks=None, tickHeight=0, scale=None, labels=['Aucune\ndouleur','Pire douleur\nimaginable'], marker=visual.Rect(win, width=0.01, height=0.1, lineColor='white', fillColor='white', units='norm'), markerColor=(255, 255, 255), markerStart=0.5, textColor='white', pos=(0, 0), stretch=2, showValue=None, showAccept=None, textSize=1.2)
            
            pos = np.random.randint(1, 10)
            ratingScale.setMarkerPos(pos)
            
            
            
            ratingScale.reset()
            # keep track of which components have finished
            eval_douleurComponents = [ratingScale, main_text, confirm_text]
            for thisComponent in eval_douleurComponents:
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
            
            # --- Run Routine "eval_douleur" ---
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
                            print(ratingScale.getRating())
                            core.wait(0.1)
                            continueRoutine=False
                            break
                
                    #print(kb.state)
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
                for thisComponent in eval_douleurComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "eval_douleur" ---
            for thisComponent in eval_douleurComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('eval_douleur.stopped', globalClock.getTime())
            # Run 'End Routine' code from slidercode
            #thisExp.addData('loca_thermode', loc)
            #thisExp.addData('condition', cond)
            #thisExp.addData('temp_trial_sent', temp_flat)
            # store data for loop_test (TrialHandler)
            loop_test.addData('ratingScale.response', ratingScale.getRating())
            loop_test.addData('ratingScale.rt', ratingScale.getRT())
            # the Routine "eval_douleur" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 16.0 repeats of 'loop_test'
        
        
        # --- Prepare to start Routine "Fin_test" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Fin_test.started', globalClock.getTime())
        key_resp_4.keys = []
        key_resp_4.rt = []
        _key_resp_4_allKeys = []
        # keep track of which components have finished
        Fin_testComponents = [text, key_resp_4]
        for thisComponent in Fin_testComponents:
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
        
        # --- Run Routine "Fin_test" ---
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
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # *key_resp_4* updates
            waitOnFlip = False
            
            # if key_resp_4 is starting this frame...
            if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_4.frameNStart = frameN  # exact frame index
                key_resp_4.tStart = t  # local t and not account for scr refresh
                key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_resp_4.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_4.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_4.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_4_allKeys.extend(theseKeys)
                if len(_key_resp_4_allKeys):
                    key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                    key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                    key_resp_4.duration = _key_resp_4_allKeys[-1].duration
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
            for thisComponent in Fin_testComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Fin_test" ---
        for thisComponent in Fin_testComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Fin_test.stopped', globalClock.getTime())
        # the Routine "Fin_test" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        loop_eval_discri = data.TrialHandler(nReps=4.0, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='loop_eval_discri')
        thisExp.addLoop(loop_eval_discri)  # add the loop to the experiment
        thisLoop_eval_discri = loop_eval_discri.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_eval_discri.rgb)
        if thisLoop_eval_discri != None:
            for paramName in thisLoop_eval_discri:
                globals()[paramName] = thisLoop_eval_discri[paramName]
        
        for thisLoop_eval_discri in loop_eval_discri:
            currentLoop = loop_eval_discri
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
            # abbreviate parameter names if possible (e.g. rgb = thisLoop_eval_discri.rgb)
            if thisLoop_eval_discri != None:
                for paramName in thisLoop_eval_discri:
                    globals()[paramName] = thisLoop_eval_discri[paramName]
            
            # --- Prepare to start Routine "instructions_eval" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('instructions_eval.started', globalClock.getTime())
            key_resp_6.keys = []
            key_resp_6.rt = []
            _key_resp_6_allKeys = []
            # keep track of which components have finished
            instructions_evalComponents = [text_7, key_resp_6]
            for thisComponent in instructions_evalComponents:
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
            
            # --- Run Routine "instructions_eval" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_7* updates
                
                # if text_7 is starting this frame...
                if text_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_7.frameNStart = frameN  # exact frame index
                    text_7.tStart = t  # local t and not account for scr refresh
                    text_7.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_7, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    text_7.status = STARTED
                    text_7.setAutoDraw(True)
                
                # if text_7 is active this frame...
                if text_7.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_6* updates
                waitOnFlip = False
                
                # if key_resp_6 is starting this frame...
                if key_resp_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_6.frameNStart = frameN  # exact frame index
                    key_resp_6.tStart = t  # local t and not account for scr refresh
                    key_resp_6.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_6, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    key_resp_6.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_6.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_6.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_6.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_6.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_6_allKeys.extend(theseKeys)
                    if len(_key_resp_6_allKeys):
                        key_resp_6.keys = _key_resp_6_allKeys[-1].name  # just the last key pressed
                        key_resp_6.rt = _key_resp_6_allKeys[-1].rt
                        key_resp_6.duration = _key_resp_6_allKeys[-1].duration
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
                for thisComponent in instructions_evalComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "instructions_eval" ---
            for thisComponent in instructions_evalComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('instructions_eval.stopped', globalClock.getTime())
            # the Routine "instructions_eval" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            loop_eval = data.TrialHandler(nReps=2.0, method='random', 
                extraInfo=expInfo, originPath=-1,
                trialList=[None],
                seed=None, name='loop_eval')
            thisExp.addLoop(loop_eval)  # add the loop to the experiment
            thisLoop_eval = loop_eval.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisLoop_eval.rgb)
            if thisLoop_eval != None:
                for paramName in thisLoop_eval:
                    globals()[paramName] = thisLoop_eval[paramName]
            
            for thisLoop_eval in loop_eval:
                currentLoop = loop_eval
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
                # abbreviate parameter names if possible (e.g. rgb = thisLoop_eval.rgb)
                if thisLoop_eval != None:
                    for paramName in thisLoop_eval:
                        globals()[paramName] = thisLoop_eval[paramName]
                
                # --- Prepare to start Routine "pause_5_EVA" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('pause_5_EVA.started', globalClock.getTime())
                # Run 'Begin Routine' code from EVA_Read
                curr_item_eval += 1
                print('curr_item_eval is ' + str(curr_item_eval))
                trial = 'trial_' + str(curr_item_eval).zfill(2)
                print('eval trial is ' + trial)
                
                cond, loc = liste_eval.loc[part, trial].split('_')
                print('eval cond and loc ' + cond + '   ' + loc)
                
                thermode_localisation = 'T : ' + str(loc) 
                # keep track of which components have finished
                pause_5_EVAComponents = [text_eva, croix_2]
                for thisComponent in pause_5_EVAComponents:
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
                
                # --- Run Routine "pause_5_EVA" ---
                routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 5.0:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *text_eva* updates
                    
                    # if text_eva is starting this frame...
                    if text_eva.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        text_eva.frameNStart = frameN  # exact frame index
                        text_eva.tStart = t  # local t and not account for scr refresh
                        text_eva.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(text_eva, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        text_eva.status = STARTED
                        text_eva.setAutoDraw(True)
                    
                    # if text_eva is active this frame...
                    if text_eva.status == STARTED:
                        # update params
                        text_eva.setText(thermode_localisation, log=False)
                    
                    # if text_eva is stopping this frame...
                    if text_eva.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > text_eva.tStartRefresh + 5-frameTolerance:
                            # keep track of stop time/frame for later
                            text_eva.tStop = t  # not accounting for scr refresh
                            text_eva.frameNStop = frameN  # exact frame index
                            # update status
                            text_eva.status = FINISHED
                            text_eva.setAutoDraw(False)
                    
                    # *croix_2* updates
                    
                    # if croix_2 is starting this frame...
                    if croix_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        croix_2.frameNStart = frameN  # exact frame index
                        croix_2.tStart = t  # local t and not account for scr refresh
                        croix_2.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(croix_2, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        croix_2.status = STARTED
                        croix_2.setAutoDraw(True)
                    
                    # if croix_2 is active this frame...
                    if croix_2.status == STARTED:
                        # update params
                        pass
                    
                    # if croix_2 is stopping this frame...
                    if croix_2.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > croix_2.tStartRefresh + 5.0-frameTolerance:
                            # keep track of stop time/frame for later
                            croix_2.tStop = t  # not accounting for scr refresh
                            croix_2.frameNStop = frameN  # exact frame index
                            # update status
                            croix_2.status = FINISHED
                            croix_2.setAutoDraw(False)
                    
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
                    for thisComponent in pause_5_EVAComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "pause_5_EVA" ---
                for thisComponent in pause_5_EVAComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('pause_5_EVA.stopped', globalClock.getTime())
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if routineForceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-5.000000)
                
                # --- Prepare to start Routine "tâche_EVA" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('tâche_EVA.started', globalClock.getTime())
                # Run 'Begin Routine' code from EVA_Trig
                port_thermode.port.write('Ue00000'.encode())
                
                
                thermode_localisation = 'T : ' + str(loc) 
                
                #
                port_thermode.set_stim(target=temp_flat, rise_rate=rise_flat, return_rate=rise_flat,
                            dur_ms=11500,
                            dur_mode='fixed_total',
                            surfaces=[1, 2, 3, 4, 5])
                
                ## Trigger thermode in other thread
                out_file = u'data/%s_%s/%s_%s_%s%s' % (expInfo['participant'], data.getDateStr(format="%Y-%m-%d"), expInfo['participant'], expName, expInfo['date'], expInfo['participant'] + '_temp_trial_eval_pla_' + str(curr_item_eval).zfill(2) + '.csv')
                stim_thread = threading.Thread(target=port_thermode.trigger_and_save_temp_rd,
                                           args=(out_file, 11500,))
                stim_thread.start()
                # keep track of which components have finished
                tâche_EVAComponents = [thermode_locali_EVA]
                for thisComponent in tâche_EVAComponents:
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
                
                # --- Run Routine "tâche_EVA" ---
                routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 12.0:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from EVA_Trig
                    if cond == 'active' :
                        polygon_1.draw()
                    else :
                        polygon_2.draw()
                    
                    # *thermode_locali_EVA* updates
                    
                    # if thermode_locali_EVA is starting this frame...
                    if thermode_locali_EVA.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        thermode_locali_EVA.frameNStart = frameN  # exact frame index
                        thermode_locali_EVA.tStart = t  # local t and not account for scr refresh
                        thermode_locali_EVA.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(thermode_locali_EVA, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'thermode_locali_EVA.started')
                        # update status
                        thermode_locali_EVA.status = STARTED
                        thermode_locali_EVA.setAutoDraw(True)
                    
                    # if thermode_locali_EVA is active this frame...
                    if thermode_locali_EVA.status == STARTED:
                        # update params
                        thermode_locali_EVA.setText(thermode_localisation, log=False)
                    
                    # if thermode_locali_EVA is stopping this frame...
                    if thermode_locali_EVA.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > thermode_locali_EVA.tStartRefresh + 12-frameTolerance:
                            # keep track of stop time/frame for later
                            thermode_locali_EVA.tStop = t  # not accounting for scr refresh
                            thermode_locali_EVA.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'thermode_locali_EVA.stopped')
                            # update status
                            thermode_locali_EVA.status = FINISHED
                            thermode_locali_EVA.setAutoDraw(False)
                    
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
                    for thisComponent in tâche_EVAComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "tâche_EVA" ---
                for thisComponent in tâche_EVAComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('tâche_EVA.stopped', globalClock.getTime())
                # Run 'End Routine' code from EVA_Trig
                
                thisExp.addData('loca_thermode', loc)
                thisExp.addData('condition', cond)
                thisExp.addData('temp_trial_sent', temp_flat)
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if routineForceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-12.000000)
                
                # --- Prepare to start Routine "eval_douleur" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('eval_douleur.started', globalClock.getTime())
                # Run 'Begin Routine' code from slidercode
                
                ratingScale = visual.RatingScale(win=win, name='ratingScale', lineColor=(255, 255, 255), low=0, high=100, precision=100, size=1, tickMarks=None, tickHeight=0, scale=None, labels=['Aucune\ndouleur','Pire douleur\nimaginable'], marker=visual.Rect(win, width=0.01, height=0.1, lineColor='white', fillColor='white', units='norm'), markerColor=(255, 255, 255), markerStart=0.5, textColor='white', pos=(0, 0), stretch=2, showValue=None, showAccept=None, textSize=1.2)
                
                pos = np.random.randint(1, 10)
                ratingScale.setMarkerPos(pos)
                
                
                
                ratingScale.reset()
                # keep track of which components have finished
                eval_douleurComponents = [ratingScale, main_text, confirm_text]
                for thisComponent in eval_douleurComponents:
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
                
                # --- Run Routine "eval_douleur" ---
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
                                print(ratingScale.getRating())
                                core.wait(0.1)
                                continueRoutine=False
                                break
                    
                        #print(kb.state)
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
                    for thisComponent in eval_douleurComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "eval_douleur" ---
                for thisComponent in eval_douleurComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('eval_douleur.stopped', globalClock.getTime())
                # Run 'End Routine' code from slidercode
                #thisExp.addData('loca_thermode', loc)
                #thisExp.addData('condition', cond)
                #thisExp.addData('temp_trial_sent', temp_flat)
                # store data for loop_eval (TrialHandler)
                loop_eval.addData('ratingScale.response', ratingScale.getRating())
                loop_eval.addData('ratingScale.rt', ratingScale.getRT())
                # the Routine "eval_douleur" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                thisExp.nextEntry()
                
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
            # completed 2.0 repeats of 'loop_eval'
            
            
            # --- Prepare to start Routine "instructions_discrim" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('instructions_discrim.started', globalClock.getTime())
            key_resp_7.keys = []
            key_resp_7.rt = []
            _key_resp_7_allKeys = []
            # keep track of which components have finished
            instructions_discrimComponents = [text_8, key_resp_7]
            for thisComponent in instructions_discrimComponents:
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
            
            # --- Run Routine "instructions_discrim" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_8* updates
                
                # if text_8 is starting this frame...
                if text_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_8.frameNStart = frameN  # exact frame index
                    text_8.tStart = t  # local t and not account for scr refresh
                    text_8.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_8, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    text_8.status = STARTED
                    text_8.setAutoDraw(True)
                
                # if text_8 is active this frame...
                if text_8.status == STARTED:
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
                for thisComponent in instructions_discrimComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "instructions_discrim" ---
            for thisComponent in instructions_discrimComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('instructions_discrim.stopped', globalClock.getTime())
            # the Routine "instructions_discrim" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # set up handler to look after randomisation of conditions etc
            loop_discri = data.TrialHandler(nReps=4.0, method='random', 
                extraInfo=expInfo, originPath=-1,
                trialList=[None],
                seed=None, name='loop_discri')
            thisExp.addLoop(loop_discri)  # add the loop to the experiment
            thisLoop_discri = loop_discri.trialList[0]  # so we can initialise stimuli with some values
            # abbreviate parameter names if possible (e.g. rgb = thisLoop_discri.rgb)
            if thisLoop_discri != None:
                for paramName in thisLoop_discri:
                    globals()[paramName] = thisLoop_discri[paramName]
            
            for thisLoop_discri in loop_discri:
                currentLoop = loop_discri
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
                # abbreviate parameter names if possible (e.g. rgb = thisLoop_discri.rgb)
                if thisLoop_discri != None:
                    for paramName in thisLoop_discri:
                        globals()[paramName] = thisLoop_discri[paramName]
                
                # --- Prepare to start Routine "pause_5_pic" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('pause_5_pic.started', globalClock.getTime())
                # Run 'Begin Routine' code from get_list_discri_2
                curr_item_discri += 1
                
                print(curr_item_discri)
                trial = 'trial_' + str(curr_item_discri).zfill(2)
                cond, pic, loc = liste_discri.loc[part, trial].split('_')
                thermode_localisation = 'T : ' + str(loc) 
                print('disc trial_pic is ' + trial)
                print('disc Pic : cond and loc ' + cond + '   ' + loc)
                
                # keep track of which components have finished
                pause_5_picComponents = [text_pic, croix_3]
                for thisComponent in pause_5_picComponents:
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
                
                # --- Run Routine "pause_5_pic" ---
                routineForceEnded = not continueRoutine
                while continueRoutine and routineTimer.getTime() < 5.0:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    
                    # *text_pic* updates
                    
                    # if text_pic is starting this frame...
                    if text_pic.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        text_pic.frameNStart = frameN  # exact frame index
                        text_pic.tStart = t  # local t and not account for scr refresh
                        text_pic.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(text_pic, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        text_pic.status = STARTED
                        text_pic.setAutoDraw(True)
                    
                    # if text_pic is active this frame...
                    if text_pic.status == STARTED:
                        # update params
                        text_pic.setText(thermode_localisation, log=False)
                    
                    # if text_pic is stopping this frame...
                    if text_pic.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > text_pic.tStartRefresh + 5-frameTolerance:
                            # keep track of stop time/frame for later
                            text_pic.tStop = t  # not accounting for scr refresh
                            text_pic.frameNStop = frameN  # exact frame index
                            # update status
                            text_pic.status = FINISHED
                            text_pic.setAutoDraw(False)
                    
                    # *croix_3* updates
                    
                    # if croix_3 is starting this frame...
                    if croix_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        croix_3.frameNStart = frameN  # exact frame index
                        croix_3.tStart = t  # local t and not account for scr refresh
                        croix_3.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(croix_3, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        croix_3.status = STARTED
                        croix_3.setAutoDraw(True)
                    
                    # if croix_3 is active this frame...
                    if croix_3.status == STARTED:
                        # update params
                        pass
                    
                    # if croix_3 is stopping this frame...
                    if croix_3.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > croix_3.tStartRefresh + 5.0-frameTolerance:
                            # keep track of stop time/frame for later
                            croix_3.tStop = t  # not accounting for scr refresh
                            croix_3.frameNStop = frameN  # exact frame index
                            # update status
                            croix_3.status = FINISHED
                            croix_3.setAutoDraw(False)
                    
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
                    for thisComponent in pause_5_picComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "pause_5_pic" ---
                for thisComponent in pause_5_picComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('pause_5_pic.stopped', globalClock.getTime())
                # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
                if routineForceEnded:
                    routineTimer.reset()
                else:
                    routineTimer.addTime(-5.000000)
                
                # --- Prepare to start Routine "tâche_PIC" ---
                continueRoutine = True
                # update component parameters for each repeat
                thisExp.addData('tâche_PIC.started', globalClock.getTime())
                # Run 'Begin Routine' code from trigger_pic
                thermode_localisation = 'T : ' + str(loc) 
                response_pic = None
                
                # Jitter duration
                dur_plateau_1 = round(random.uniform(3.25, 4.75), 2)
                
                # Plateau 2 is difference 10.95 - others
                dur_plateau_2 = round(10 - (dur_plateau_1 + 1.2), 2)
                
                # Detection cue always start at 3.5
                start1 = 3
                # Responses start after rise to plateau + plateau + pic (1.2)
                start2 = 8
                
                
                # Convert to string
                dur_send = str(int(dur_plateau_1*100))
                dur_send2 = str(int(dur_plateau_2*100))
                
                #Pic or not?
                
                if pic == 'pic-present' :
                    # Set thermode
                    port_thermode.port.write('Ue11111'.encode())
                    port_thermode.set_rd_plateau(temp_plateau=temp_flat_str,
                                      temp_pic=pic_str,
                                      dur_plateau_1_10ms=dur_send,
                                      dur_plateau_2_10ms=dur_send2)
                else :
                    # Set thermode
                    port_thermode.port.write('Ue11111'.encode())
                    port_thermode.set_rd_plateau(temp_plateau=temp_flat_str,
                                      temp_pic=temp_flat_str,
                                      dur_plateau_1_10ms=dur_send,
                                      dur_plateau_2_10ms=dur_send2)
                
                # Trigger thermode in other thread
                out_file = u'data/%s_%s/%s_%s_%s%s' % (expInfo['participant'], data.getDateStr(format="%Y-%m-%d"), expInfo['participant'], expName, expInfo['date'], expInfo['participant'] + '_temp_trial_pic_' + str(curr_item_discri).zfill(2) + '.csv')
                stim_thread = threading.Thread(target=port_thermode.trigger_and_save_temp_rd,
                                     args=(out_file, 11500,))
                stim_thread.start()
                
                
                
                
                
                
                
                thermode_locali_pic.setText(thermode_localisation)
                discrimin_resp.keys = []
                discrimin_resp.rt = []
                _discrimin_resp_allKeys = []
                responsel.setText(loca_textl)
                # keep track of which components have finished
                tâche_PICComponents = [thermode_locali_pic, discrimin_resp, responsel, percu_or, responser]
                for thisComponent in tâche_PICComponents:
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
                
                # --- Run Routine "tâche_PIC" ---
                routineForceEnded = not continueRoutine
                while continueRoutine:
                    # get current time
                    t = routineTimer.getTime()
                    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame
                    # Run 'Each Frame' code from trigger_pic
                    #Polygon normal avant signal
                    if (t <= dur_plateau_1) and cond == 'active': 
                        polygon_1.draw()
                    if (t <= dur_plateau_1) and cond == 'inactive':
                        polygon_2.draw()
                    
                    #Remplir en noir pour signal
                    if(t >= start1) and cond == 'active' and (t <= start2): 
                        polygon_1.draw()
                        polygon_1_black.draw()
                    if (t >= start1) and cond == 'inactive' and (t <= start2): 
                        polygon_2.draw()
                        polygon_2_black.draw()
                    # Check response and change color answer
                    if discrimin_resp.status == STARTED and len(discrimin_resp.keys) != 0:
                        if discrimin_resp.keys[-1] == 'm':
                            responsel.color = (0, 0, 0)
                            response_pic = detectr
                        elif discrimin_resp.keys[-1] == 'n':
                            responser.color = (0, 0, 0)
                            response_pic = detectl
                    
                    #if (t >= 7.90) :
                        #win.flip()
                    
                    # *thermode_locali_pic* updates
                    
                    # if thermode_locali_pic is starting this frame...
                    if thermode_locali_pic.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                        # keep track of start time/frame for later
                        thermode_locali_pic.frameNStart = frameN  # exact frame index
                        thermode_locali_pic.tStart = t  # local t and not account for scr refresh
                        thermode_locali_pic.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(thermode_locali_pic, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'thermode_locali_pic.started')
                        # update status
                        thermode_locali_pic.status = STARTED
                        thermode_locali_pic.setAutoDraw(True)
                    
                    # if thermode_locali_pic is active this frame...
                    if thermode_locali_pic.status == STARTED:
                        # update params
                        pass
                    
                    # if thermode_locali_pic is stopping this frame...
                    if thermode_locali_pic.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > thermode_locali_pic.tStartRefresh + 10.95-frameTolerance:
                            # keep track of stop time/frame for later
                            thermode_locali_pic.tStop = t  # not accounting for scr refresh
                            thermode_locali_pic.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'thermode_locali_pic.stopped')
                            # update status
                            thermode_locali_pic.status = FINISHED
                            thermode_locali_pic.setAutoDraw(False)
                    
                    # *discrimin_resp* updates
                    waitOnFlip = False
                    
                    # if discrimin_resp is starting this frame...
                    if discrimin_resp.status == NOT_STARTED and tThisFlip >= start2-frameTolerance:
                        # keep track of start time/frame for later
                        discrimin_resp.frameNStart = frameN  # exact frame index
                        discrimin_resp.tStart = t  # local t and not account for scr refresh
                        discrimin_resp.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(discrimin_resp, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        discrimin_resp.status = STARTED
                        # keyboard checking is just starting
                        waitOnFlip = True
                        win.callOnFlip(discrimin_resp.clock.reset)  # t=0 on next screen flip
                        win.callOnFlip(discrimin_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                    
                    # if discrimin_resp is stopping this frame...
                    if discrimin_resp.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > discrimin_resp.tStartRefresh + 3-frameTolerance:
                            # keep track of stop time/frame for later
                            discrimin_resp.tStop = t  # not accounting for scr refresh
                            discrimin_resp.frameNStop = frameN  # exact frame index
                            # update status
                            discrimin_resp.status = FINISHED
                            discrimin_resp.status = FINISHED
                    if discrimin_resp.status == STARTED and not waitOnFlip:
                        theseKeys = discrimin_resp.getKeys(keyList=['n', 'm'], ignoreKeys=["escape"], waitRelease=False)
                        _discrimin_resp_allKeys.extend(theseKeys)
                        if len(_discrimin_resp_allKeys):
                            discrimin_resp.keys = _discrimin_resp_allKeys[0].name  # just the first key pressed
                            discrimin_resp.rt = _discrimin_resp_allKeys[0].rt
                            discrimin_resp.duration = _discrimin_resp_allKeys[0].duration
                    
                    # *responsel* updates
                    
                    # if responsel is starting this frame...
                    if responsel.status == NOT_STARTED and tThisFlip >= start2-frameTolerance:
                        # keep track of start time/frame for later
                        responsel.frameNStart = frameN  # exact frame index
                        responsel.tStart = t  # local t and not account for scr refresh
                        responsel.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(responsel, 'tStartRefresh')  # time at next scr refresh
                        # update status
                        responsel.status = STARTED
                        responsel.setAutoDraw(True)
                    
                    # if responsel is active this frame...
                    if responsel.status == STARTED:
                        # update params
                        pass
                    
                    # if responsel is stopping this frame...
                    if responsel.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > responsel.tStartRefresh + 3-frameTolerance:
                            # keep track of stop time/frame for later
                            responsel.tStop = t  # not accounting for scr refresh
                            responsel.frameNStop = frameN  # exact frame index
                            # update status
                            responsel.status = FINISHED
                            responsel.setAutoDraw(False)
                    
                    # *percu_or* updates
                    
                    # if percu_or is starting this frame...
                    if percu_or.status == NOT_STARTED and tThisFlip >= start2-frameTolerance:
                        # keep track of start time/frame for later
                        percu_or.frameNStart = frameN  # exact frame index
                        percu_or.tStart = t  # local t and not account for scr refresh
                        percu_or.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(percu_or, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'percu_or.started')
                        # update status
                        percu_or.status = STARTED
                        percu_or.setAutoDraw(True)
                    
                    # if percu_or is active this frame...
                    if percu_or.status == STARTED:
                        # update params
                        pass
                    
                    # if percu_or is stopping this frame...
                    if percu_or.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > percu_or.tStartRefresh + 3-frameTolerance:
                            # keep track of stop time/frame for later
                            percu_or.tStop = t  # not accounting for scr refresh
                            percu_or.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'percu_or.stopped')
                            # update status
                            percu_or.status = FINISHED
                            percu_or.setAutoDraw(False)
                    
                    # *responser* updates
                    
                    # if responser is starting this frame...
                    if responser.status == NOT_STARTED and tThisFlip >= start2-frameTolerance:
                        # keep track of start time/frame for later
                        responser.frameNStart = frameN  # exact frame index
                        responser.tStart = t  # local t and not account for scr refresh
                        responser.tStartRefresh = tThisFlipGlobal  # on global time
                        win.timeOnFlip(responser, 'tStartRefresh')  # time at next scr refresh
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'responser.started')
                        # update status
                        responser.status = STARTED
                        responser.setAutoDraw(True)
                    
                    # if responser is active this frame...
                    if responser.status == STARTED:
                        # update params
                        pass
                    
                    # if responser is stopping this frame...
                    if responser.status == STARTED:
                        # is it time to stop? (based on global clock, using actual start)
                        if tThisFlipGlobal > responser.tStartRefresh + 3-frameTolerance:
                            # keep track of stop time/frame for later
                            responser.tStop = t  # not accounting for scr refresh
                            responser.frameNStop = frameN  # exact frame index
                            # add timestamp to datafile
                            thisExp.timestampOnFlip(win, 'responser.stopped')
                            # update status
                            responser.status = FINISHED
                            responser.setAutoDraw(False)
                    
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
                    for thisComponent in tâche_PICComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine = True
                            break  # at least one component has not yet finished
                    
                    # refresh the screen
                    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                        win.flip()
                
                # --- Ending Routine "tâche_PIC" ---
                for thisComponent in tâche_PICComponents:
                    if hasattr(thisComponent, "setAutoDraw"):
                        thisComponent.setAutoDraw(False)
                thisExp.addData('tâche_PIC.stopped', globalClock.getTime())
                # Run 'End Routine' code from trigger_pic
                trials.addData('dur_plateau1', dur_plateau_1)
                trials.addData('dur_plateau2', dur_plateau_2)
                trials.addData('dur_plateau1_sent', dur_send)
                trials.addData('dur_plateau2_sent', dur_send2)
                trials.addData('start_plateau_2_resp', start2)
                trials.addData('dur_pic', 1.2)
                trials.addData('dur_rise', 0.75)
                
                thisExp.addData('loca_thermode', loc)
                thisExp.addData('condition', cond)
                thisExp.addData('pic_presence', pic)
                trials.addData('pic_response', response_pic)
                # Reset color
                responsel.color = 'white'
                responser.color = 'white'
                # check responses
                if discrimin_resp.keys in ['', [], None]:  # No response was made
                    discrimin_resp.keys = None
                loop_discri.addData('discrimin_resp.keys',discrimin_resp.keys)
                if discrimin_resp.keys != None:  # we had a response
                    loop_discri.addData('discrimin_resp.rt', discrimin_resp.rt)
                    loop_discri.addData('discrimin_resp.duration', discrimin_resp.duration)
                # the Routine "tâche_PIC" was not non-slip safe, so reset the non-slip timer
                routineTimer.reset()
                thisExp.nextEntry()
                
                if thisSession is not None:
                    # if running in a Session with a Liaison client, send data up to now
                    thisSession.sendExperimentData()
            # completed 4.0 repeats of 'loop_discri'
            
        # completed 4.0 repeats of 'loop_eval_discri'
        
        
        # --- Prepare to start Routine "break_entre_bloc" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('break_entre_bloc.started', globalClock.getTime())
        # Run 'Begin Routine' code from Clock
        stimclock = core.Clock()
        starttime = stimclock.getTime()
        key_resp_2.keys = []
        key_resp_2.rt = []
        _key_resp_2_allKeys = []
        # keep track of which components have finished
        break_entre_blocComponents = [text_3, key_resp_2]
        for thisComponent in break_entre_blocComponents:
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
        
        # --- Run Routine "break_entre_bloc" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from Clock
            remaining = str(60 - round(stimclock.getTime()- starttime))
            text_clock = '      Bloc Terminé, pause '   +  remaining  + ' sec        ' + '           Appuyer sur p lorsque vous serez prêt pour le prochain bloc.    '
            
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
                text_3.setText(text_clock, log=False)
            
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
            for thisComponent in break_entre_blocComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "break_entre_bloc" ---
        for thisComponent in break_entre_blocComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('break_entre_bloc.stopped', globalClock.getTime())
        # the Routine "break_entre_bloc" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 2.0 repeats of 'trials'
    
    
    # --- Prepare to start Routine "merci" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('merci.started', globalClock.getTime())
    # keep track of which components have finished
    merciComponents = [text_6]
    for thisComponent in merciComponents:
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
    
    # --- Run Routine "merci" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 10.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_6* updates
        
        # if text_6 is starting this frame...
        if text_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_6.frameNStart = frameN  # exact frame index
            text_6.tStart = t  # local t and not account for scr refresh
            text_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_6, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_6.status = STARTED
            text_6.setAutoDraw(True)
        
        # if text_6 is active this frame...
        if text_6.status == STARTED:
            # update params
            pass
        
        # if text_6 is stopping this frame...
        if text_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_6.tStartRefresh + 10.0-frameTolerance:
                # keep track of stop time/frame for later
                text_6.tStop = t  # not accounting for scr refresh
                text_6.frameNStop = frameN  # exact frame index
                # update status
                text_6.status = FINISHED
                text_6.setAutoDraw(False)
        
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
        for thisComponent in merciComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "merci" ---
    for thisComponent in merciComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('merci.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-10.000000)
    # Run 'End Experiment' code from trigger_pic
    
    
    
    
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
